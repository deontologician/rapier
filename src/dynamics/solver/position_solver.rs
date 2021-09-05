use super::AnyJointPositionConstraint;
use crate::data::{ComponentSet, ComponentSetMut};
use crate::dynamics::solver::generic_position_constraint::AnyGenericPositionConstraint;
use crate::dynamics::{
    solver::AnyPositionConstraint, ArticulationSet, IntegrationParameters, IslandManager,
    RigidBodyIds, RigidBodyMassProps, RigidBodyPosition,
};
use crate::math::{Isometry, Real};
use na::DVector;

pub(crate) struct PositionSolver {
    positions: Vec<Isometry<Real>>,
}

impl PositionSolver {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
        }
    }

    pub fn solve<Bodies>(
        &mut self,
        island_id: usize,
        params: &IntegrationParameters,
        islands: &IslandManager,
        bodies: &mut Bodies,
        multibodies: &mut ArticulationSet,
        contact_constraints: &[AnyPositionConstraint],
        generic_contact_constraints: &[AnyGenericPositionConstraint],
        joint_constraints: &[AnyJointPositionConstraint],
        workspace: &mut DVector<Real>, // Must contain at least `4 * max(ndofs)` elements.
    ) where
        Bodies: ComponentSet<RigidBodyIds>
            + ComponentSetMut<RigidBodyPosition>
            + ComponentSetMut<RigidBodyMassProps>,
    {
        if contact_constraints.is_empty()
            && generic_contact_constraints.is_empty()
            && joint_constraints.is_empty()
        {
            // Nothing to do.
            return;
        }

        self.positions.clear();
        self.positions
            .extend(islands.active_island(island_id).iter().map(|h| {
                let poss: &RigidBodyPosition = bodies.index(h.0);
                poss.next_position
            }));

        for _ in 0..params.max_position_iterations {
            for constraint in joint_constraints {
                constraint.solve(params, &mut self.positions)
            }

            for constraint in contact_constraints {
                constraint.solve(params, &mut self.positions)
            }

            for constraint in generic_contact_constraints {
                constraint.solve(params, multibodies, bodies, &mut self.positions, workspace)
            }
        }

        for handle in islands.active_island(island_id) {
            if multibodies.rigid_body_link(*handle).is_none() {
                let ids: &RigidBodyIds = bodies.index(handle.0);
                let next_pos = &self.positions[ids.active_set_offset];
                bodies.map_mut_internal(handle.0, |poss: &mut RigidBodyPosition| {
                    poss.next_position = *next_pos
                });
            }
        }
    }
}
