use super::{
    AnyJointVelocityConstraint, InteractionGroups, VelocityConstraint, VelocityGroundConstraint,
};
#[cfg(feature = "simd-is-enabled")]
use super::{
    WPositionConstraint, WPositionGroundConstraint, WVelocityConstraint, WVelocityGroundConstraint,
};
use crate::data::ComponentSet;
use crate::dynamics::solver::categorization::{categorize_contacts, categorize_joints};
use crate::dynamics::solver::{
    AnyJointPositionConstraint, AnyPositionConstraint, GenericVelocityConstraint,
    PositionConstraint, PositionGroundConstraint,
};
use crate::dynamics::{
    solver::AnyVelocityConstraint, IntegrationParameters, JointGraphEdge, JointIndex, MultibodySet,
    RigidBodyIds, RigidBodyMassProps, RigidBodyPosition, RigidBodyType,
};
use crate::dynamics::{IslandManager, RigidBodyVelocity};
use crate::geometry::{ContactManifold, ContactManifoldIndex};
use crate::math::Real;
#[cfg(feature = "simd-is-enabled")]
use crate::math::SIMD_WIDTH;
use na::DVector;

pub(crate) struct SolverConstraints<VelocityConstraint, PositionConstraint, GenVelocityConstraint> {
    pub generic_jacobians: DVector<Real>,
    pub not_ground_interactions: Vec<usize>,
    pub ground_interactions: Vec<usize>,
    pub generic_interactions: Vec<usize>,
    pub interaction_groups: InteractionGroups,
    pub ground_interaction_groups: InteractionGroups,
    pub velocity_constraints: Vec<VelocityConstraint>,
    pub position_constraints: Vec<PositionConstraint>,
    pub generic_velocity_constraints: Vec<GenVelocityConstraint>,
}

impl<VelocityConstraint, PositionConstraint, GenVelocityConstraint>
    SolverConstraints<VelocityConstraint, PositionConstraint, GenVelocityConstraint>
{
    pub fn new() -> Self {
        Self {
            generic_jacobians: DVector::zeros(0),
            not_ground_interactions: Vec::new(),
            ground_interactions: Vec::new(),
            generic_interactions: Vec::new(),
            interaction_groups: InteractionGroups::new(),
            ground_interaction_groups: InteractionGroups::new(),
            velocity_constraints: Vec::new(),
            position_constraints: Vec::new(),
            generic_velocity_constraints: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.not_ground_interactions.clear();
        self.ground_interactions.clear();
        self.generic_interactions.clear();
        self.interaction_groups.clear();
        self.ground_interaction_groups.clear();
        self.velocity_constraints.clear();
        self.position_constraints.clear();
    }
}

impl SolverConstraints<AnyVelocityConstraint, AnyPositionConstraint, GenericVelocityConstraint> {
    pub fn init_constraint_groups<Bodies>(
        &mut self,
        island_id: usize,
        islands: &IslandManager,
        bodies: &Bodies,
        manifolds: &[&mut ContactManifold],
        manifold_indices: &[ContactManifoldIndex],
    ) where
        Bodies: ComponentSet<RigidBodyType> + ComponentSet<RigidBodyIds>,
    {
        self.not_ground_interactions.clear();
        self.ground_interactions.clear();
        self.generic_interactions.clear();

        categorize_contacts(
            bodies,
            manifolds,
            manifold_indices,
            &mut self.ground_interactions,
            &mut self.not_ground_interactions,
            &mut self.generic_interactions,
        );

        self.interaction_groups.clear_groups();
        self.interaction_groups.group_manifolds(
            island_id,
            islands,
            bodies,
            manifolds,
            &self.not_ground_interactions,
        );

        self.ground_interaction_groups.clear_groups();
        self.ground_interaction_groups.group_manifolds(
            island_id,
            islands,
            bodies,
            manifolds,
            &self.ground_interactions,
        );

        // NOTE: uncomment this do disable SIMD contact resolution.
        //        self.interaction_groups
        //            .nongrouped_interactions
        //            .append(&mut self.interaction_groups.grouped_interactions);
        //        self.ground_interaction_groups
        //            .nongrouped_interactions
        //            .append(&mut self.ground_interaction_groups.grouped_interactions);
    }

    pub fn init<Bodies>(
        &mut self,
        island_id: usize,
        params: &IntegrationParameters,
        islands: &IslandManager,
        bodies: &Bodies,
        multibodies: &MultibodySet,
        manifolds: &[&mut ContactManifold],
        manifold_indices: &[ContactManifoldIndex],
    ) where
        Bodies: ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyVelocity>
            + ComponentSet<RigidBodyMassProps>
            + ComponentSet<RigidBodyIds>
            + ComponentSet<RigidBodyType>,
    {
        self.velocity_constraints.clear();
        self.position_constraints.clear();
        self.generic_velocity_constraints.clear();

        self.init_constraint_groups(island_id, islands, bodies, manifolds, manifold_indices);

        #[cfg(feature = "simd-is-enabled")]
        {
            self.compute_grouped_constraints(params, bodies, manifolds);
        }
        self.compute_nongrouped_constraints(params, bodies, manifolds);
        self.compute_generic_constraints(params, bodies, multibodies, manifolds);

        #[cfg(feature = "simd-is-enabled")]
        {
            self.compute_grouped_ground_constraints(params, bodies, manifolds);
        }
        self.compute_nongrouped_ground_constraints(params, bodies, manifolds);
    }

    #[cfg(feature = "simd-is-enabled")]
    fn compute_grouped_constraints<Bodies>(
        &mut self,
        params: &IntegrationParameters,
        bodies: &Bodies,
        manifolds_all: &[&mut ContactManifold],
    ) where
        Bodies: ComponentSet<RigidBodyVelocity>
            + ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyMassProps>
            + ComponentSet<RigidBodyIds>,
    {
        for manifolds_i in self
            .interaction_groups
            .grouped_interactions
            .chunks_exact(SIMD_WIDTH)
        {
            let manifold_id = gather![|ii| manifolds_i[ii]];
            let manifolds = gather![|ii| &*manifolds_all[manifolds_i[ii]]];
            WVelocityConstraint::generate(
                params,
                manifold_id,
                manifolds,
                bodies,
                &mut self.velocity_constraints,
                true,
            );
            WPositionConstraint::generate(
                params,
                manifolds,
                bodies,
                &mut self.position_constraints,
                true,
            );
        }
    }

    fn compute_nongrouped_constraints<Bodies>(
        &mut self,
        params: &IntegrationParameters,
        bodies: &Bodies,
        manifolds_all: &[&mut ContactManifold],
    ) where
        Bodies: ComponentSet<RigidBodyVelocity>
            + ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyMassProps>
            + ComponentSet<RigidBodyIds>,
    {
        for manifold_i in &self.interaction_groups.nongrouped_interactions {
            let manifold = &manifolds_all[*manifold_i];
            VelocityConstraint::generate(
                params,
                *manifold_i,
                manifold,
                bodies,
                &mut self.velocity_constraints,
                true,
            );
            PositionConstraint::generate(
                params,
                manifold,
                bodies,
                &mut self.position_constraints,
                true,
            );
        }
    }

    fn compute_generic_constraints<Bodies>(
        &mut self,
        params: &IntegrationParameters,
        bodies: &Bodies,
        multibodies: &MultibodySet,
        manifolds_all: &[&mut ContactManifold],
    ) where
        Bodies: ComponentSet<RigidBodyVelocity>
            + ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyMassProps>
            + ComponentSet<RigidBodyIds>
            + ComponentSet<RigidBodyType>,
    {
        let mut jacobian_id = 0;
        for manifold_i in &self.generic_interactions {
            let manifold = &manifolds_all[*manifold_i];
            GenericVelocityConstraint::generate(
                params,
                *manifold_i,
                manifold,
                bodies,
                multibodies,
                &mut self.generic_velocity_constraints,
                &mut self.generic_jacobians,
                &mut jacobian_id,
                true,
            );
            // PositionConstraint::generate(
            //     params,
            //     manifold,
            //     bodies,
            //     &mut self.position_constraints,
            //     true,
            // );
        }
    }

    #[cfg(feature = "simd-is-enabled")]
    fn compute_grouped_ground_constraints<Bodies>(
        &mut self,
        params: &IntegrationParameters,
        bodies: &Bodies,
        manifolds_all: &[&mut ContactManifold],
    ) where
        Bodies: ComponentSet<RigidBodyIds>
            + ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyVelocity>
            + ComponentSet<RigidBodyMassProps>,
    {
        for manifolds_i in self
            .ground_interaction_groups
            .grouped_interactions
            .chunks_exact(SIMD_WIDTH)
        {
            let manifold_id = gather![|ii| manifolds_i[ii]];
            let manifolds = gather![|ii| &*manifolds_all[manifolds_i[ii]]];
            WVelocityGroundConstraint::generate(
                params,
                manifold_id,
                manifolds,
                bodies,
                &mut self.velocity_constraints,
                true,
            );
            WPositionGroundConstraint::generate(
                params,
                manifolds,
                bodies,
                &mut self.position_constraints,
                true,
            );
        }
    }

    fn compute_nongrouped_ground_constraints<Bodies>(
        &mut self,
        params: &IntegrationParameters,
        bodies: &Bodies,
        manifolds_all: &[&mut ContactManifold],
    ) where
        Bodies: ComponentSet<RigidBodyIds>
            + ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyVelocity>
            + ComponentSet<RigidBodyMassProps>,
    {
        for manifold_i in &self.ground_interaction_groups.nongrouped_interactions {
            let manifold = &manifolds_all[*manifold_i];
            VelocityGroundConstraint::generate(
                params,
                *manifold_i,
                manifold,
                bodies,
                &mut self.velocity_constraints,
                true,
            );
            PositionGroundConstraint::generate(
                params,
                manifold,
                bodies,
                &mut self.position_constraints,
                true,
            )
        }
    }
}

impl SolverConstraints<AnyJointVelocityConstraint, AnyJointPositionConstraint, ()> {
    pub fn init<Bodies>(
        &mut self,
        island_id: usize,
        params: &IntegrationParameters,
        islands: &IslandManager,
        bodies: &Bodies,
        joints: &[JointGraphEdge],
        joint_constraint_indices: &[JointIndex],
    ) where
        Bodies: ComponentSet<RigidBodyType>
            + ComponentSet<RigidBodyIds>
            + ComponentSet<RigidBodyVelocity>
            + ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyMassProps>,
    {
        // Generate constraints for joints.
        self.not_ground_interactions.clear();
        self.ground_interactions.clear();
        categorize_joints(
            bodies,
            joints,
            joint_constraint_indices,
            &mut self.ground_interactions,
            &mut self.not_ground_interactions,
        );

        self.velocity_constraints.clear();
        self.position_constraints.clear();

        self.interaction_groups.clear_groups();
        self.interaction_groups.group_joints(
            island_id,
            islands,
            bodies,
            joints,
            &self.not_ground_interactions,
        );

        self.ground_interaction_groups.clear_groups();
        self.ground_interaction_groups.group_joints(
            island_id,
            islands,
            bodies,
            joints,
            &self.ground_interactions,
        );
        // NOTE: uncomment this do disable SIMD joint resolution.
        // self.interaction_groups
        //     .nongrouped_interactions
        //     .append(&mut self.interaction_groups.grouped_interactions);
        // self.ground_interaction_groups
        //     .nongrouped_interactions
        //     .append(&mut self.ground_interaction_groups.grouped_interactions);

        self.compute_nongrouped_joint_ground_constraints(params, bodies, joints);
        #[cfg(feature = "simd-is-enabled")]
        {
            self.compute_grouped_joint_ground_constraints(params, bodies, joints);
        }
        self.compute_nongrouped_joint_constraints(params, bodies, joints);
        #[cfg(feature = "simd-is-enabled")]
        {
            self.compute_grouped_joint_constraints(params, bodies, joints);
        }
    }

    fn compute_nongrouped_joint_ground_constraints<Bodies>(
        &mut self,
        params: &IntegrationParameters,
        bodies: &Bodies,
        joints_all: &[JointGraphEdge],
    ) where
        Bodies: ComponentSet<RigidBodyType>
            + ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyMassProps>
            + ComponentSet<RigidBodyVelocity>
            + ComponentSet<RigidBodyIds>,
    {
        for joint_i in &self.ground_interaction_groups.nongrouped_interactions {
            let joint = &joints_all[*joint_i].weight;
            let vel_constraint =
                AnyJointVelocityConstraint::from_joint_ground(params, *joint_i, joint, bodies);
            self.velocity_constraints.push(vel_constraint);
            let pos_constraint = AnyJointPositionConstraint::from_joint_ground(joint, bodies);
            self.position_constraints.push(pos_constraint);
        }
    }

    #[cfg(feature = "simd-is-enabled")]
    fn compute_grouped_joint_ground_constraints<Bodies>(
        &mut self,
        params: &IntegrationParameters,
        bodies: &Bodies,
        joints_all: &[JointGraphEdge],
    ) where
        Bodies: ComponentSet<RigidBodyType>
            + ComponentSet<RigidBodyVelocity>
            + ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyMassProps>
            + ComponentSet<RigidBodyIds>,
    {
        for joints_i in self
            .ground_interaction_groups
            .grouped_interactions
            .chunks_exact(SIMD_WIDTH)
        {
            let joints_id = gather![|ii| joints_i[ii]];
            let joints = gather![|ii| &joints_all[joints_i[ii]].weight];
            let vel_constraint = AnyJointVelocityConstraint::from_wide_joint_ground(
                params, joints_id, joints, bodies,
            );
            self.velocity_constraints.push(vel_constraint);

            let pos_constraint = AnyJointPositionConstraint::from_wide_joint_ground(joints, bodies);
            self.position_constraints.push(pos_constraint);
        }
    }

    fn compute_nongrouped_joint_constraints<Bodies>(
        &mut self,
        params: &IntegrationParameters,
        bodies: &Bodies,
        joints_all: &[JointGraphEdge],
    ) where
        Bodies: ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyVelocity>
            + ComponentSet<RigidBodyMassProps>
            + ComponentSet<RigidBodyIds>,
    {
        for joint_i in &self.interaction_groups.nongrouped_interactions {
            let joint = &joints_all[*joint_i].weight;
            let vel_constraint =
                AnyJointVelocityConstraint::from_joint(params, *joint_i, joint, bodies);
            self.velocity_constraints.push(vel_constraint);
            let pos_constraint = AnyJointPositionConstraint::from_joint(joint, bodies);
            self.position_constraints.push(pos_constraint);
        }
    }

    #[cfg(feature = "simd-is-enabled")]
    fn compute_grouped_joint_constraints<Bodies>(
        &mut self,
        params: &IntegrationParameters,
        bodies: &Bodies,
        joints_all: &[JointGraphEdge],
    ) where
        Bodies: ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyVelocity>
            + ComponentSet<RigidBodyMassProps>
            + ComponentSet<RigidBodyIds>,
    {
        for joints_i in self
            .interaction_groups
            .grouped_interactions
            .chunks_exact(SIMD_WIDTH)
        {
            let joints_id = gather![|ii| joints_i[ii]];
            let joints = gather![|ii| &joints_all[joints_i[ii]].weight];
            let vel_constraint =
                AnyJointVelocityConstraint::from_wide_joint(params, joints_id, joints, bodies);
            self.velocity_constraints.push(vel_constraint);

            let pos_constraint = AnyJointPositionConstraint::from_wide_joint(joints, bodies);
            self.position_constraints.push(pos_constraint);
        }
    }
}
