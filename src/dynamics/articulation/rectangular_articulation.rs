use na::{self, DVectorSliceMut, Isometry3, Translation3, Unit, Vector3};

use crate::dynamics::{
    Articulation, IntegrationParameters, PrismaticArticulation, RigidBodyVelocity,
};
use crate::math::{JacobianSliceMut, Real};

/// A articulation that allows two translational degrees of freedom.
#[derive(Copy, Clone, Debug)]
pub struct RectangularArticulation {
    prism1: PrismaticArticulation,
    prism2: PrismaticArticulation,
}

impl RectangularArticulation {
    /// Creates a new rectangular articulation allowing relative translations along the two provided axes.
    ///
    /// Both axes are expressed in the local coordinate frame on the attached multibody links.
    pub fn new(
        axis1: Unit<Vector3<Real>>,
        axis2: Unit<Vector3<Real>>,
        offset1: Real,
        offset2: Real,
    ) -> Self {
        RectangularArticulation {
            prism1: PrismaticArticulation::new(axis1, offset1),
            prism2: PrismaticArticulation::new(axis2, offset2),
        }
    }
}

impl Articulation for RectangularArticulation {
    #[inline]
    fn ndofs(&self) -> usize {
        2
    }

    fn body_to_parent(
        &self,
        parent_shift: &Vector3<Real>,
        body_shift: &Vector3<Real>,
    ) -> Isometry3<Real> {
        let t = Translation3::from(parent_shift - body_shift)
            * self.prism1.translation()
            * self.prism2.translation();
        Isometry3::from_parts(t, na::one())
    }

    fn update_jacobians(&mut self, body_shift: &Vector3<Real>, vels: &[Real]) {
        self.prism1.update_jacobians(body_shift, vels);
        self.prism2.update_jacobians(body_shift, &[vels[1]]);
    }

    fn jacobian(&self, transform: &Isometry3<Real>, out: &mut JacobianSliceMut<Real>) {
        self.prism1.jacobian(transform, &mut out.columns_mut(0, 1));
        self.prism2.jacobian(transform, &mut out.columns_mut(1, 1));
    }

    fn jacobian_dot(&self, _: &Isometry3<Real>, _: &mut JacobianSliceMut<Real>) {}

    fn jacobian_dot_veldiff_mul_coordinates(
        &self,
        _: &Isometry3<Real>,
        _: &[Real],
        _: &mut JacobianSliceMut<Real>,
    ) {
    }

    fn jacobian_mul_coordinates(&self, vels: &[Real]) -> RigidBodyVelocity {
        self.prism1.jacobian_mul_coordinates(vels)
            + self.prism2.jacobian_mul_coordinates(&[vels[1]])
    }

    fn jacobian_dot_mul_coordinates(&self, _: &[Real]) -> RigidBodyVelocity {
        RigidBodyVelocity::zero()
    }

    fn default_damping(&self, out: &mut DVectorSliceMut<Real>) {
        self.prism1.default_damping(&mut out.rows_mut(0, 1));
        self.prism2.default_damping(&mut out.rows_mut(1, 1));
    }

    fn integrate(&mut self, dt: Real, vels: &[Real]) {
        self.prism1.integrate(dt, vels);
        self.prism2.integrate(dt, &[vels[1]]);
    }

    fn apply_displacement(&mut self, disp: &[Real]) {
        self.prism1.apply_displacement(disp);
        self.prism2.apply_displacement(&[disp[1]]);
    }

    #[inline]
    fn clone(&self) -> Box<dyn Articulation> {
        Box::new(*self)
    }

    /*
    fn num_velocity_constraints(&self) -> usize {
        self.prism1.num_velocity_constraints() + self.prism2.num_velocity_constraints()
    }

    fn velocity_constraints(
        &self,
        parameters: &IntegrationParameters,
        multibody: &Multibody<Real>,
        link: &MultibodyLink<Real>,
        assembly_id: usize,
        dof_id: usize,
        ext_vels: &[Real],
        ground_j_id: &mut usize,
        jacobians: &mut [Real],
        constraints: &mut ConstraintSet<Real, (), (), usize>,
    ) {
        self.prism1.velocity_constraints(
            parameters,
            multibody,
            link,
            assembly_id,
            dof_id,
            ext_vels,
            ground_j_id,
            jacobians,
            constraints,
        );
        self.prism2.velocity_constraints(
            parameters,
            multibody,
            link,
            assembly_id,
            dof_id + 1,
            ext_vels,
            ground_j_id,
            jacobians,
            constraints,
        );
    }

    fn num_position_constraints(&self) -> usize {
        // NOTE: we don't test if constraints exist to simplify indexing.
        2
    }

    fn position_constraint(
        &self,
        i: usize,
        multibody: &Multibody<Real>,
        link: &MultibodyLink<Real>,
        handle: BodyPartHandle<()>,
        dof_id: usize,
        jacobians: &mut [Real],
    ) -> Option<GenericNonlinearConstraint<Real, ()>> {
        if i == 0 {
            self.prism1
                .position_constraint(0, multibody, link, handle, dof_id, jacobians)
        } else {
            self.prism2
                .position_constraint(0, multibody, link, handle, dof_id + 1, jacobians)
        }
    }
     */
}

prismatic_motor_limit_methods_1!(RectangularArticulation, prism1);
prismatic_motor_limit_methods_2!(RectangularArticulation, prism2);
