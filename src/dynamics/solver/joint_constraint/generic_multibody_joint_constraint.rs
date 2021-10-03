use crate::dynamics::solver::DeltaVel;
use crate::dynamics::{
    BallJoint, IntegrationParameters, JointGraphEdge, JointIndex, JointParams, Multibody,
    RigidBodyIds, RigidBodyMassProps, RigidBodyPosition, RigidBodyVelocity,
};
use crate::math::{
    AngVector, AngularInertia, Dim, Real, Rotation, SdpMatrix, Vector, DIM, SPATIAL_DIM,
};
use crate::utils::{WAngularInertia, WCross, WCrossMatrix, WDot};
use na::{
    matrix, Const, DVector, DVectorSlice, DVectorSliceMut, Dynamic, MatrixSlice, SMatrix, SVector,
    U1,
};

type JacSlice<'a, const IMP_DIM: usize> = MatrixSlice<'a, Real, Dynamic, Const<IMP_DIM>>;

#[derive(Debug)]
pub(crate) struct GenericMultibodyJointConstraint<const IMP_DIM: usize> {
    is_rigid_body1: bool, // TODO: merge the two bools into a single bit mask
    is_rigid_body2: bool,
    mj_lambda1: usize,
    mj_lambda2: usize,

    ndofs1: usize,
    j_id1: usize,
    ndofs2: usize,
    j_id2: usize,

    joint_id: JointIndex,

    rhs: SVector<Real, IMP_DIM>,
    impulse: SVector<Real, IMP_DIM>,
    inv_lhs: SMatrix<Real, IMP_DIM, IMP_DIM>,
}

impl GenericMultibodyJointConstraint<3> {
    pub fn ball_constraint(
        params: &IntegrationParameters,
        joint_id: JointIndex,
        rb1: (
            &RigidBodyPosition,
            &RigidBodyVelocity,
            &RigidBodyMassProps,
            &RigidBodyIds,
        ),
        rb2: (
            &RigidBodyPosition,
            &RigidBodyVelocity,
            &RigidBodyMassProps,
            &RigidBodyIds,
        ),
        mb1: Option<(&Multibody, usize)>,
        mb2: Option<(&Multibody, usize)>,
        j_id: &mut usize,
        jacobians: &mut DVector<Real>,
        joint: &BallJoint,
    ) -> Self {
        let (rb_pos1, rb_vels1, rb_mprops1, rb_ids1) = rb1;
        let (rb_pos2, rb_vels2, rb_mprops2, rb_ids2) = rb2;

        let anchor_world1 = rb_pos1.position * joint.local_anchor1;
        let anchor_world2 = rb_pos2.position * joint.local_anchor2;
        let anchor1 = anchor_world1 - rb_mprops1.world_com;
        let anchor2 = anchor_world2 - rb_mprops2.world_com;

        let vel1 = rb_vels1.linvel + rb_vels1.angvel.gcross(anchor1);
        let vel2 = rb_vels2.linvel + rb_vels2.angvel.gcross(anchor2);

        let rhs = (vel2 - vel1) * params.velocity_solve_fraction
            + (anchor_world2 - anchor_world1) * params.velocity_based_erp_inv_dt();

        let j_id1 = *j_id;
        let joint_j1 = joint.jacobian_matrix(-1.0, &anchor1);
        let lhs1 = if let Some((mb1, link_id1)) = mb1 {
            mb1.fill_jacobians_generic(link_id1, &joint_j1, j_id, jacobians)
        } else {
            rb_mprops1.fill_jacobians_generic(&joint_j1, j_id, jacobians)
        };

        let j_id2 = *j_id;
        let joint_j2 = joint.jacobian_matrix(1.0, &anchor2);
        let lhs2 = if let Some((mb2, link_id2)) = mb2 {
            mb2.fill_jacobians_generic(link_id2, &joint_j2, j_id, jacobians)
        } else {
            rb_mprops2.fill_jacobians_generic(&joint_j2, j_id, jacobians)
        };

        let lhs = lhs1 + lhs2;

        let inv_lhs = lhs.try_inverse().expect("Found singular system");

        GenericMultibodyJointConstraint {
            is_rigid_body1: mb1.is_none(),
            is_rigid_body2: mb2.is_none(),
            ndofs1: mb1.map(|m| m.0.ndofs()).unwrap_or(SPATIAL_DIM),
            ndofs2: mb2.map(|m| m.0.ndofs()).unwrap_or(SPATIAL_DIM),
            j_id1,
            j_id2,
            joint_id,
            mj_lambda1: mb1
                .map(|m| m.0.solver_id)
                .unwrap_or(rb_ids1.active_set_offset),
            mj_lambda2: mb2
                .map(|m| m.0.solver_id)
                .unwrap_or(rb_ids2.active_set_offset),
            impulse: joint.impulse * params.warmstart_coeff,
            rhs,
            inv_lhs,
        }
    }
}

impl<const IMP_DIM: usize> GenericMultibodyJointConstraint<IMP_DIM> {
    fn wj_id1(&self) -> usize {
        self.j_id1 + self.ndofs1 * DIM
    }

    fn wj_id2(&self) -> usize {
        self.j_id2 + self.ndofs2 * DIM
    }

    fn mj_lambda1<'a>(
        &self,
        mj_lambdas: &'a [DeltaVel<Real>],
        generic_mj_lambdas: &'a DVector<Real>,
    ) -> DVectorSlice<'a, Real> {
        if self.is_rigid_body1 {
            mj_lambdas[self.mj_lambda1].as_vector_slice()
        } else {
            generic_mj_lambdas.rows(self.mj_lambda1, self.ndofs1)
        }
    }

    fn mj_lambda1_mut<'a>(
        &self,
        mj_lambdas: &'a mut [DeltaVel<Real>],
        generic_mj_lambdas: &'a mut DVector<Real>,
    ) -> DVectorSliceMut<'a, Real> {
        if self.is_rigid_body1 {
            mj_lambdas[self.mj_lambda1].as_vector_slice_mut()
        } else {
            generic_mj_lambdas.rows_mut(self.mj_lambda1, self.ndofs1)
        }
    }

    fn mj_lambda2<'a>(
        &self,
        mj_lambdas: &'a [DeltaVel<Real>],
        generic_mj_lambdas: &'a DVector<Real>,
    ) -> DVectorSlice<'a, Real> {
        if self.is_rigid_body2 {
            mj_lambdas[self.mj_lambda2].as_vector_slice()
        } else {
            generic_mj_lambdas.rows(self.mj_lambda2, self.ndofs2)
        }
    }

    fn mj_lambda2_mut<'a>(
        &self,
        mj_lambdas: &'a mut [DeltaVel<Real>],
        generic_mj_lambdas: &'a mut DVector<Real>,
    ) -> DVectorSliceMut<'a, Real> {
        if self.is_rigid_body2 {
            mj_lambdas[self.mj_lambda2].as_vector_slice_mut()
        } else {
            generic_mj_lambdas.rows_mut(self.mj_lambda2, self.ndofs2)
        }
    }

    pub fn warmstart(
        &self,
        jacobians: &DVector<Real>,
        mj_lambdas: &mut [DeltaVel<Real>],
        generic_mj_lambdas: &mut DVector<Real>,
    ) {
        let jacobians = jacobians.as_slice();

        // First body.
        let mut mj_lambda = self.mj_lambda1_mut(mj_lambdas, generic_mj_lambdas);
        let wj = JacSlice::<IMP_DIM>::from_slice(&jacobians[self.wj_id1()..], self.ndofs1);
        mj_lambda.gemv(1.0, &wj, &self.impulse, 1.0);

        // Second body.
        let mut mj_lambda = self.mj_lambda2_mut(mj_lambdas, generic_mj_lambdas);
        let wj = JacSlice::<IMP_DIM>::from_slice(&jacobians[self.wj_id2()..], self.ndofs2);
        mj_lambda.gemv(1.0, &wj, &self.impulse, 1.0);
    }

    pub fn solve(
        &mut self,
        jacobians: &DVector<Real>,
        mj_lambdas: &mut [DeltaVel<Real>],
        generic_mj_lambdas: &mut DVector<Real>,
    ) {
        let jacobians = jacobians.as_slice();
        let mj_lambda1 = self.mj_lambda1(mj_lambdas, generic_mj_lambdas);
        let j1 = JacSlice::<IMP_DIM>::from_slice(&jacobians[self.j_id1..], self.ndofs1);
        let vel1 = j1.tr_mul(&mj_lambda1);

        let mj_lambda2 = self.mj_lambda2(mj_lambdas, generic_mj_lambdas);
        let j2 = JacSlice::<IMP_DIM>::from_slice(&jacobians[self.j_id2..], self.ndofs2);
        let vel2 = j2.tr_mul(&mj_lambda2);

        let dvel = vel1 + vel2 + self.rhs;

        let impulse = self.inv_lhs * dvel;
        self.impulse += impulse;

        let mut mj_lambda1 = self.mj_lambda1_mut(mj_lambdas, generic_mj_lambdas);
        let wj1 = JacSlice::<IMP_DIM>::from_slice(&jacobians[self.wj_id1()..], self.ndofs1);
        mj_lambda1.gemv(1.0, &wj1, &impulse, 1.0);

        let mut mj_lambda2 = self.mj_lambda2_mut(mj_lambdas, generic_mj_lambdas);
        let wj2 = JacSlice::<IMP_DIM>::from_slice(&jacobians[self.wj_id2()..], self.ndofs2);
        mj_lambda2.gemv(1.0, &wj2, &impulse, 1.0);
    }

    pub fn writeback_impulses(&self, joints_all: &mut [JointGraphEdge]) {
        let joint = &mut joints_all[self.joint_id].weight;

        match &mut joint.params {
            JointParams::BallJoint(ball) => ball.impulse.copy_from_slice(self.impulse.as_slice()),
            _ => todo!(),
        }
    }
}
