use super::AnyJointVelocityConstraint;
use crate::data::{BundleSet, ComponentSet, ComponentSetMut};
use crate::dynamics::solver::GenericVelocityConstraint;
use crate::dynamics::{
    solver::{AnyVelocityConstraint, DeltaVel},
    IntegrationParameters, JointGraphEdge, MultibodySet, RigidBodyForces, RigidBodyVelocity,
};
use crate::dynamics::{IslandManager, RigidBodyIds, RigidBodyMassProps};
use crate::geometry::ContactManifold;
use crate::math::Real;
use crate::utils::WAngularInertia;
use na::DVector;

pub(crate) struct VelocitySolver {
    pub mj_lambdas: Vec<DeltaVel<Real>>,
    pub generic_mj_lambdas: DVector<Real>,
}

impl VelocitySolver {
    pub fn new() -> Self {
        Self {
            mj_lambdas: Vec::new(),
            generic_mj_lambdas: DVector::zeros(0),
        }
    }

    pub fn solve<Bodies>(
        &mut self,
        island_id: usize,
        params: &IntegrationParameters,
        islands: &IslandManager,
        bodies: &mut Bodies,
        multibodies: &mut MultibodySet,
        manifolds_all: &mut [&mut ContactManifold],
        joints_all: &mut [JointGraphEdge],
        contact_constraints: &mut [AnyVelocityConstraint],
        generic_contact_constraints: &mut [GenericVelocityConstraint],
        generic_contact_jacobians: &DVector<Real>,
        joint_constraints: &mut [AnyJointVelocityConstraint],
    ) where
        Bodies: ComponentSet<RigidBodyForces>
            + ComponentSet<RigidBodyIds>
            + ComponentSetMut<RigidBodyVelocity>
            + ComponentSet<RigidBodyMassProps>,
    {
        self.mj_lambdas.clear();
        self.mj_lambdas
            .resize(islands.active_island(island_id).len(), DeltaVel::zero());

        let total_multibodies_ndofs = multibodies.multibodies.iter().map(|m| m.1.ndofs()).sum();
        self.generic_mj_lambdas = DVector::zeros(total_multibodies_ndofs);

        // Initialize delta-velocities (`mj_lambdas`) with external forces (gravity etc):
        for handle in islands.active_island(island_id) {
            let (ids, mprops, forces): (&RigidBodyIds, &RigidBodyMassProps, &RigidBodyForces) =
                bodies.index_bundle(handle.0);

            let dvel = &mut self.mj_lambdas[ids.active_set_offset];

            // NOTE: `dvel.angular` is actually storing angular velocity delta multiplied
            //       by the square root of the inertia tensor:
            dvel.angular += mprops.effective_world_inv_inertia_sqrt * forces.torque * params.dt;
            dvel.linear += forces.force * (mprops.effective_inv_mass * params.dt);
        }

        for (_, multibody) in multibodies.multibodies.iter_mut() {
            let mut mj_lambdas = self
                .generic_mj_lambdas
                .rows_mut(multibody.solver_id, multibody.ndofs());
            mj_lambdas.axpy(params.dt, &multibody.accelerations, 0.0);
        }

        /*
         * Warmstart constraints.
         */
        for constraint in &*joint_constraints {
            constraint.warmstart(&mut self.mj_lambdas[..]);
        }

        for constraint in &*contact_constraints {
            constraint.warmstart(&mut self.mj_lambdas[..]);
        }

        for constraint in &*generic_contact_constraints {
            constraint.warmstart(
                generic_contact_jacobians,
                &mut self.mj_lambdas[..],
                &mut self.generic_mj_lambdas,
            );
        }

        /*
         * Solve constraints.
         */
        for _ in 0..params.max_velocity_iterations {
            for constraint in &mut *joint_constraints {
                constraint.solve(&mut self.mj_lambdas[..]);
            }

            for constraint in &mut *contact_constraints {
                constraint.solve(&mut self.mj_lambdas[..]);
            }

            for constraint in &mut *generic_contact_constraints {
                constraint.solve(
                    generic_contact_jacobians,
                    &mut self.mj_lambdas[..],
                    &mut self.generic_mj_lambdas,
                );
            }
        }

        // Update velocities.
        for handle in islands.active_island(island_id) {
            let (ids, mprops): (&RigidBodyIds, &RigidBodyMassProps) = bodies.index_bundle(handle.0);

            let dvel = self.mj_lambdas[ids.active_set_offset];
            let dangvel = mprops
                .effective_world_inv_inertia_sqrt
                .transform_vector(dvel.angular);

            bodies.map_mut_internal(handle.0, |vels| {
                vels.linvel += dvel.linear;
                vels.angvel += dangvel;
            });
        }

        for (_, multibody) in multibodies.multibodies.iter_mut() {
            let mut mj_lambdas = self
                .generic_mj_lambdas
                .rows(multibody.solver_id, multibody.ndofs());
            multibody.velocities += mj_lambdas;
            multibody.integrate(params.dt); // TODO: this shouldn't be done here.
        }

        // Write impulses back into the manifold structures.
        for constraint in &*joint_constraints {
            constraint.writeback_impulses(joints_all);
        }

        for constraint in &*contact_constraints {
            constraint.writeback_impulses(manifolds_all);
        }

        for constraint in &*generic_contact_constraints {
            constraint.writeback_impulses(manifolds_all);
        }
    }
}
