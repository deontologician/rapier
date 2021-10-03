use super::{PositionSolver, VelocitySolver};
use crate::counters::Counters;
use crate::data::{BundleSet, ComponentSet, ComponentSetMut};
use crate::dynamics::solver::generic_position_constraint::AnyGenericPositionConstraint;
use crate::dynamics::solver::{
    AnyJointPositionConstraint, AnyJointVelocityConstraint, AnyPositionConstraint,
    AnyVelocityConstraint, GenericVelocityConstraint, SolverConstraints,
};
use crate::dynamics::{
    IntegrationParameters, JointGraphEdge, JointIndex, RigidBodyDamping, RigidBodyForces,
    RigidBodyIds, RigidBodyMassProps, RigidBodyPosition, RigidBodyType,
};
use crate::dynamics::{IslandManager, RigidBodyVelocity};
use crate::geometry::{ContactManifold, ContactManifoldIndex};
use crate::prelude::ArticulationSet;

pub struct IslandSolver {
    contact_constraints: SolverConstraints<
        AnyVelocityConstraint,
        AnyPositionConstraint,
        GenericVelocityConstraint,
        AnyGenericPositionConstraint,
    >,
    joint_constraints:
        SolverConstraints<AnyJointVelocityConstraint, AnyJointPositionConstraint, (), ()>,
    velocity_solver: VelocitySolver,
    position_solver: PositionSolver,
}

impl Default for IslandSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl IslandSolver {
    pub fn new() -> Self {
        Self {
            contact_constraints: SolverConstraints::new(),
            joint_constraints: SolverConstraints::new(),
            velocity_solver: VelocitySolver::new(),
            position_solver: PositionSolver::new(),
        }
    }

    // NOTE: the position resolution MUST happen after the the velocity
    //       resolution.
    pub fn solve_position_constraints<Bodies>(
        &mut self,
        island_id: usize,
        islands: &IslandManager,
        counters: &mut Counters,
        params: &IntegrationParameters,
        bodies: &mut Bodies,
        multibodies: &mut ArticulationSet,
    ) where
        Bodies: ComponentSet<RigidBodyIds>
            + ComponentSetMut<RigidBodyPosition>
            + ComponentSetMut<RigidBodyMassProps>,
    {
        counters.solver.position_resolution_time.resume();
        self.position_solver.solve(
            island_id,
            params,
            islands,
            bodies,
            multibodies,
            &self.contact_constraints.position_constraints,
            &self.contact_constraints.generic_position_constraints,
            &self.joint_constraints.position_constraints,
            // NOTE: the jacobian vector of velocity constraints is
            //       large enough to serve as a workspace for the position solver.
            //       It’s fine for us to mess up with this vector because we already
            //       used these jacobians during the velocity iterations which must
            //       happen before the position iterations.
            &mut self.contact_constraints.generic_jacobians,
        );
        counters.solver.position_resolution_time.pause();
    }

    pub fn init_constraints_and_solve_velocity_constraints<Bodies>(
        &mut self,
        island_id: usize,
        counters: &mut Counters,
        params: &IntegrationParameters,
        islands: &IslandManager,
        bodies: &mut Bodies,
        manifolds: &mut [&mut ContactManifold],
        manifold_indices: &[ContactManifoldIndex],
        joints: &mut [JointGraphEdge],
        joint_indices: &[JointIndex],
        multibodies: &mut ArticulationSet,
    ) where
        Bodies: ComponentSet<RigidBodyForces>
            + ComponentSetMut<RigidBodyPosition>
            + ComponentSetMut<RigidBodyVelocity>
            + ComponentSet<RigidBodyMassProps>
            + ComponentSet<RigidBodyDamping>
            + ComponentSet<RigidBodyIds>
            + ComponentSet<RigidBodyType>,
    {
        let has_constraints = manifold_indices.len() != 0 || joint_indices.len() != 0;

        if has_constraints {
            // Init the solver id for multibodies.
            // We need that for building the constraints.
            let mut solver_id = 0;
            for (_, multibody) in multibodies.multibodies.iter_mut() {
                multibody.solver_id = solver_id;
                solver_id += multibody.ndofs();
            }

            counters.solver.velocity_assembly_time.resume();
            self.contact_constraints.init(
                island_id,
                params,
                islands,
                bodies,
                multibodies,
                manifolds,
                manifold_indices,
            );
            self.joint_constraints.init(
                island_id,
                params,
                islands,
                bodies,
                multibodies,
                joints,
                joint_indices,
            );
            counters.solver.velocity_assembly_time.pause();

            counters.solver.velocity_resolution_time.resume();
            self.velocity_solver.solve(
                island_id,
                params,
                islands,
                bodies,
                multibodies,
                manifolds,
                joints,
                &mut self.contact_constraints.velocity_constraints,
                &mut self.contact_constraints.generic_velocity_constraints,
                &self.contact_constraints.generic_jacobians,
                &mut self.joint_constraints.velocity_constraints,
                &self.joint_constraints.generic_jacobians,
            );
            counters.solver.velocity_resolution_time.pause();

            counters.solver.velocity_update_time.resume();

            for handle in islands.active_island(island_id) {
                if let Some(link) = multibodies.rigid_body_link(*handle).copied() {
                    let multibody = multibodies
                        .get_multibody_mut_internal(link.multibody)
                        .unwrap();

                    if link.id == 0 || link.id == 1 && !multibody.root_is_dynamic {
                        multibody.integrate(params.dt);
                    }
                } else {
                    let (poss, vels, damping, mprops): (
                        &RigidBodyPosition,
                        &RigidBodyVelocity,
                        &RigidBodyDamping,
                        &RigidBodyMassProps,
                    ) = bodies.index_bundle(handle.0);

                    let mut new_poss = *poss;
                    let new_vels = vels.apply_damping(params.dt, damping);
                    new_poss.next_position =
                        vels.integrate(params.dt, &poss.position, &mprops.local_mprops.local_com);

                    bodies.set_internal(handle.0, new_vels);
                    bodies.set_internal(handle.0, new_poss);
                }
            }

            counters.solver.velocity_update_time.pause();
        } else {
            self.contact_constraints.clear();
            self.joint_constraints.clear();
            counters.solver.velocity_update_time.resume();

            for handle in islands.active_island(island_id) {
                if let Some(link) = multibodies.rigid_body_link(*handle).copied() {
                    let multibody = multibodies
                        .get_multibody_mut_internal(link.multibody)
                        .unwrap();

                    if link.id == 0 || link.id == 1 && !multibody.root_is_dynamic {
                        let accels = &multibody.accelerations;
                        multibody.velocities.axpy(params.dt, accels, 1.0);
                        multibody.integrate(params.dt);
                    }
                } else {
                    // Since we didn't run the velocity solver we need to integrate the accelerations here
                    let (poss, vels, forces, damping, mprops): (
                        &RigidBodyPosition,
                        &RigidBodyVelocity,
                        &RigidBodyForces,
                        &RigidBodyDamping,
                        &RigidBodyMassProps,
                    ) = bodies.index_bundle(handle.0);

                    let mut new_poss = *poss;
                    let new_vels = forces
                        .integrate(params.dt, vels, mprops)
                        .apply_damping(params.dt, &damping);
                    new_poss.next_position =
                        vels.integrate(params.dt, &poss.position, &mprops.local_mprops.local_com);

                    bodies.set_internal(handle.0, new_vels);
                    bodies.set_internal(handle.0, new_poss);
                }
            }
            counters.solver.velocity_update_time.pause();
        }
    }
}
