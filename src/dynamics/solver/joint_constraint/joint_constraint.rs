use super::{
    BallVelocityConstraint, BallVelocityGroundConstraint, FixedVelocityConstraint,
    FixedVelocityGroundConstraint, PrismaticVelocityConstraint, PrismaticVelocityGroundConstraint,
};
#[cfg(feature = "dim3")]
use super::{RevoluteVelocityConstraint, RevoluteVelocityGroundConstraint};
#[cfg(feature = "simd-is-enabled")]
use super::{
    WBallVelocityConstraint, WBallVelocityGroundConstraint, WFixedVelocityConstraint,
    WFixedVelocityGroundConstraint, WPrismaticVelocityConstraint,
    WPrismaticVelocityGroundConstraint,
};
#[cfg(feature = "dim3")]
#[cfg(feature = "simd-is-enabled")]
use super::{WRevoluteVelocityConstraint, WRevoluteVelocityGroundConstraint};
// use crate::dynamics::solver::joint_constraint::generic_velocity_constraint::{
//     GenericVelocityConstraint, GenericVelocityGroundConstraint,
// };
use crate::data::{BundleSet, ComponentSet};
use crate::dynamics::solver::joint_constraint::generic_multibody_joint_constraint::GenericMultibodyJointConstraint;
use crate::dynamics::solver::{DeltaVel, GenericVelocityConstraint};
use crate::dynamics::{
    IntegrationParameters, Joint, JointGraphEdge, JointIndex, JointParams, RigidBodyIds,
    RigidBodyMassProps, RigidBodyPosition, RigidBodyType, RigidBodyVelocity,
};
#[cfg(feature = "simd-is-enabled")]
use crate::math::SIMD_WIDTH;
use crate::math::{Real, SPATIAL_DIM};
use crate::prelude::ArticulationSet;
use na::DVector;

pub(crate) enum AnyJointVelocityConstraint {
    GenericConstraint3Dof(GenericMultibodyJointConstraint<3>),
    BallConstraint(BallVelocityConstraint),
    BallGroundConstraint(BallVelocityGroundConstraint),
    #[cfg(feature = "simd-is-enabled")]
    WBallConstraint(WBallVelocityConstraint),
    #[cfg(feature = "simd-is-enabled")]
    WBallGroundConstraint(WBallVelocityGroundConstraint),
    FixedConstraint(FixedVelocityConstraint),
    FixedGroundConstraint(FixedVelocityGroundConstraint),
    #[cfg(feature = "simd-is-enabled")]
    WFixedConstraint(WFixedVelocityConstraint),
    #[cfg(feature = "simd-is-enabled")]
    WFixedGroundConstraint(WFixedVelocityGroundConstraint),
    // GenericConstraint(GenericVelocityConstraint),
    // GenericGroundConstraint(GenericVelocityGroundConstraint),
    // #[cfg(feature = "simd-is-enabled")]
    // WGenericConstraint(WGenericVelocityConstraint),
    // #[cfg(feature = "simd-is-enabled")]
    // WGenericGroundConstraint(WGenericVelocityGroundConstraint),
    PrismaticConstraint(PrismaticVelocityConstraint),
    PrismaticGroundConstraint(PrismaticVelocityGroundConstraint),
    #[cfg(feature = "simd-is-enabled")]
    WPrismaticConstraint(WPrismaticVelocityConstraint),
    #[cfg(feature = "simd-is-enabled")]
    WPrismaticGroundConstraint(WPrismaticVelocityGroundConstraint),
    #[cfg(feature = "dim3")]
    RevoluteConstraint(RevoluteVelocityConstraint),
    #[cfg(feature = "dim3")]
    RevoluteGroundConstraint(RevoluteVelocityGroundConstraint),
    #[cfg(feature = "dim3")]
    #[cfg(feature = "simd-is-enabled")]
    WRevoluteConstraint(WRevoluteVelocityConstraint),
    #[cfg(feature = "dim3")]
    #[cfg(feature = "simd-is-enabled")]
    WRevoluteGroundConstraint(WRevoluteVelocityGroundConstraint),
    #[allow(dead_code)] // The Empty variant is only used with parallel code.
    Empty,
}

impl AnyJointVelocityConstraint {
    #[cfg(feature = "parallel")]
    pub fn num_active_constraints(_: &Joint) -> usize {
        1
    }

    pub fn from_joint<Bodies>(
        params: &IntegrationParameters,
        joint_id: JointIndex,
        joint: &Joint,
        bodies: &Bodies,
        multibodies: &ArticulationSet,
        j_id: &mut usize,
        jacobians: &mut DVector<Real>,
    ) -> Self
    where
        Bodies: ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyVelocity>
            + ComponentSet<RigidBodyMassProps>
            + ComponentSet<RigidBodyIds>,
    {
        let rb1 = (
            bodies.index(joint.body1.0),
            bodies.index(joint.body1.0),
            bodies.index(joint.body1.0),
            bodies.index(joint.body1.0),
        );
        let rb2 = (
            bodies.index(joint.body2.0),
            bodies.index(joint.body2.0),
            bodies.index(joint.body2.0),
            bodies.index(joint.body2.0),
        );
        let mb1 = multibodies
            .rigid_body_link(joint.body1)
            .map(|link| (&multibodies[link.multibody], link.id));
        let mb2 = multibodies
            .rigid_body_link(joint.body2)
            .map(|link| (&multibodies[link.multibody], link.id));

        if mb1.is_some() || mb2.is_some() {
            let multibodies_ndof = mb1.map(|m| m.0.ndofs()).unwrap_or(SPATIAL_DIM)
                + mb2.map(|m| m.0.ndofs()).unwrap_or(SPATIAL_DIM);
            // For each solver contact we generate up to SPATIAL_DIM constraints, and each
            // constraints appends the multibodies jacobian and weighted jacobians.
            // Also note that for joints, the rigid-bodies will also add their jacobians
            // to the generic DVector.
            // TODO: is this count correct when we take both motors and limits into account?
            let required_jacobian_len = *j_id + multibodies_ndof * 2 * SPATIAL_DIM;

            if jacobians.nrows() < required_jacobian_len {
                jacobians.resize_vertically_mut(required_jacobian_len, 0.0);
            }

            match &joint.params {
                JointParams::BallJoint(p) => AnyJointVelocityConstraint::GenericConstraint3Dof(
                    GenericMultibodyJointConstraint::ball_constraint(
                        params, joint_id, rb1, rb2, mb1, mb2, j_id, jacobians, p,
                    ),
                ),
                JointParams::FixedJoint(p) => todo!(),
                JointParams::PrismaticJoint(p) => todo!(),
                #[cfg(feature = "dim3")]
                JointParams::RevoluteJoint(p) => todo!(),
            }
        } else {
            match &joint.params {
                JointParams::BallJoint(p) => {
                    AnyJointVelocityConstraint::BallConstraint(BallVelocityConstraint::from_params(
                        params, joint_id, rb1, rb2, mb1, mb2, j_id, jacobians, p,
                    ))
                }
                JointParams::FixedJoint(p) => AnyJointVelocityConstraint::FixedConstraint(
                    FixedVelocityConstraint::from_params(params, joint_id, rb1, rb2, p),
                ),
                JointParams::PrismaticJoint(p) => AnyJointVelocityConstraint::PrismaticConstraint(
                    PrismaticVelocityConstraint::from_params(params, joint_id, rb1, rb2, p),
                ),
                // JointParams::GenericJoint(p) => AnyJointVelocityConstraint::GenericConstraint(
                //     GenericVelocityConstraint::from_params(params, joint_id, rb1, rb2, p),
                // ),
                #[cfg(feature = "dim3")]
                JointParams::RevoluteJoint(p) => AnyJointVelocityConstraint::RevoluteConstraint(
                    RevoluteVelocityConstraint::from_params(params, joint_id, rb1, rb2, p),
                ),
            }
        }
    }

    #[cfg(feature = "simd-is-enabled")]
    pub fn from_wide_joint<Bodies>(
        params: &IntegrationParameters,
        joint_id: [JointIndex; SIMD_WIDTH],
        joints: [&Joint; SIMD_WIDTH],
        bodies: &Bodies,
    ) -> Self
    where
        Bodies: ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyVelocity>
            + ComponentSet<RigidBodyMassProps>
            + ComponentSet<RigidBodyIds>,
    {
        let rbs1 = (
            gather![|ii| bodies.index(joints[ii].body1.0)],
            gather![|ii| bodies.index(joints[ii].body1.0)],
            gather![|ii| bodies.index(joints[ii].body1.0)],
            gather![|ii| bodies.index(joints[ii].body1.0)],
        );
        let rbs2 = (
            gather![|ii| bodies.index(joints[ii].body2.0)],
            gather![|ii| bodies.index(joints[ii].body2.0)],
            gather![|ii| bodies.index(joints[ii].body2.0)],
            gather![|ii| bodies.index(joints[ii].body2.0)],
        );

        match &joints[0].params {
            JointParams::BallJoint(_) => {
                let joints = gather![|ii| joints[ii].params.as_ball_joint().unwrap()];
                AnyJointVelocityConstraint::WBallConstraint(WBallVelocityConstraint::from_params(
                    params, joint_id, rbs1, rbs2, joints,
                ))
            }
            JointParams::FixedJoint(_) => {
                let joints = gather![|ii| joints[ii].params.as_fixed_joint().unwrap()];
                AnyJointVelocityConstraint::WFixedConstraint(WFixedVelocityConstraint::from_params(
                    params, joint_id, rbs1, rbs2, joints,
                ))
            }
            // JointParams::GenericJoint(_) => {
            //     let joints = gather![|ii| joints[ii].params.as_generic_joint().unwrap()];
            //     AnyJointVelocityConstraint::WGenericConstraint(
            //         WGenericVelocityConstraint::from_params(params, joint_id, rbs1, rbs2, joints),
            //     )
            // }
            JointParams::PrismaticJoint(_) => {
                let joints = gather![|ii| joints[ii].params.as_prismatic_joint().unwrap()];
                AnyJointVelocityConstraint::WPrismaticConstraint(
                    WPrismaticVelocityConstraint::from_params(params, joint_id, rbs1, rbs2, joints),
                )
            }
            #[cfg(feature = "dim3")]
            JointParams::RevoluteJoint(_) => {
                let joints = gather![|ii| joints[ii].params.as_revolute_joint().unwrap()];
                AnyJointVelocityConstraint::WRevoluteConstraint(
                    WRevoluteVelocityConstraint::from_params(params, joint_id, rbs1, rbs2, joints),
                )
            }
        }
    }

    pub fn from_joint_ground<Bodies>(
        params: &IntegrationParameters,
        joint_id: JointIndex,
        joint: &Joint,
        bodies: &Bodies,
    ) -> Self
    where
        Bodies: ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyType>
            + ComponentSet<RigidBodyVelocity>
            + ComponentSet<RigidBodyMassProps>
            + ComponentSet<RigidBodyIds>,
    {
        let mut handle1 = joint.body1;
        let mut handle2 = joint.body2;
        let status2: &RigidBodyType = bodies.index(handle2.0);
        let flipped = !status2.is_dynamic();

        if flipped {
            std::mem::swap(&mut handle1, &mut handle2);
        }

        let rb1 = bodies.index_bundle(handle1.0);
        let rb2 = bodies.index_bundle(handle2.0);

        match &joint.params {
            JointParams::BallJoint(p) => AnyJointVelocityConstraint::BallGroundConstraint(
                BallVelocityGroundConstraint::from_params(params, joint_id, rb1, rb2, p, flipped),
            ),
            JointParams::FixedJoint(p) => AnyJointVelocityConstraint::FixedGroundConstraint(
                FixedVelocityGroundConstraint::from_params(params, joint_id, rb1, rb2, p, flipped),
            ),
            // JointParams::GenericJoint(p) => AnyJointVelocityConstraint::GenericGroundConstraint(
            //     GenericVelocityGroundConstraint::from_params(
            //         params, joint_id, rb1, rb2, p, flipped,
            //     ),
            // ),
            JointParams::PrismaticJoint(p) => {
                AnyJointVelocityConstraint::PrismaticGroundConstraint(
                    PrismaticVelocityGroundConstraint::from_params(
                        params, joint_id, rb1, rb2, p, flipped,
                    ),
                )
            }
            #[cfg(feature = "dim3")]
            JointParams::RevoluteJoint(p) => RevoluteVelocityGroundConstraint::from_params(
                params, joint_id, rb1, rb2, p, flipped,
            ),
        }
    }

    #[cfg(feature = "simd-is-enabled")]
    pub fn from_wide_joint_ground<Bodies>(
        params: &IntegrationParameters,
        joint_id: [JointIndex; SIMD_WIDTH],
        joints: [&Joint; SIMD_WIDTH],
        bodies: &Bodies,
    ) -> Self
    where
        Bodies: ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyType>
            + ComponentSet<RigidBodyVelocity>
            + ComponentSet<RigidBodyMassProps>
            + ComponentSet<RigidBodyIds>,
    {
        let mut handles1 = gather![|ii| joints[ii].body1];
        let mut handles2 = gather![|ii| joints[ii].body2];
        let status2: [&RigidBodyType; SIMD_WIDTH] = gather![|ii| bodies.index(handles2[ii].0)];
        let mut flipped = [false; SIMD_WIDTH];

        for ii in 0..SIMD_WIDTH {
            if !status2[ii].is_dynamic() {
                std::mem::swap(&mut handles1[ii], &mut handles2[ii]);
                flipped[ii] = true;
            }
        }

        let rbs1 = (
            gather![|ii| bodies.index(handles1[ii].0)],
            gather![|ii| bodies.index(handles1[ii].0)],
            gather![|ii| bodies.index(handles1[ii].0)],
        );
        let rbs2 = (
            gather![|ii| bodies.index(handles2[ii].0)],
            gather![|ii| bodies.index(handles2[ii].0)],
            gather![|ii| bodies.index(handles2[ii].0)],
            gather![|ii| bodies.index(handles2[ii].0)],
        );

        match &joints[0].params {
            JointParams::BallJoint(_) => {
                let joints = gather![|ii| joints[ii].params.as_ball_joint().unwrap()];
                AnyJointVelocityConstraint::WBallGroundConstraint(
                    WBallVelocityGroundConstraint::from_params(
                        params, joint_id, rbs1, rbs2, joints, flipped,
                    ),
                )
            }
            JointParams::FixedJoint(_) => {
                let joints = gather![|ii| joints[ii].params.as_fixed_joint().unwrap()];
                AnyJointVelocityConstraint::WFixedGroundConstraint(
                    WFixedVelocityGroundConstraint::from_params(
                        params, joint_id, rbs1, rbs2, joints, flipped,
                    ),
                )
            }
            // JointParams::GenericJoint(_) => {
            //     let joints = gather![|ii| joints[ii].params.as_generic_joint().unwrap()];
            //     AnyJointVelocityConstraint::WGenericGroundConstraint(
            //         WGenericVelocityGroundConstraint::from_params(
            //             params, joint_id, rbs1, rbs2, joints, flipped,
            //         ),
            //     )
            // }
            JointParams::PrismaticJoint(_) => {
                let joints = gather![|ii| joints[ii].params.as_prismatic_joint().unwrap()];
                AnyJointVelocityConstraint::WPrismaticGroundConstraint(
                    WPrismaticVelocityGroundConstraint::from_params(
                        params, joint_id, rbs1, rbs2, joints, flipped,
                    ),
                )
            }
            #[cfg(feature = "dim3")]
            JointParams::RevoluteJoint(_) => {
                let joints = gather![|ii| joints[ii].params.as_revolute_joint().unwrap()];
                AnyJointVelocityConstraint::WRevoluteGroundConstraint(
                    WRevoluteVelocityGroundConstraint::from_params(
                        params, joint_id, rbs1, rbs2, joints, flipped,
                    ),
                )
            }
        }
    }

    pub fn warmstart(
        &self,
        jacobians: &DVector<Real>,
        mj_lambdas: &mut [DeltaVel<Real>],
        generic_mj_lambdas: &mut DVector<Real>,
    ) {
        match self {
            AnyJointVelocityConstraint::GenericConstraint3Dof(c) => {
                c.warmstart(jacobians, mj_lambdas, generic_mj_lambdas)
            }
            AnyJointVelocityConstraint::BallConstraint(c) => c.warmstart(mj_lambdas),
            AnyJointVelocityConstraint::BallGroundConstraint(c) => c.warmstart(mj_lambdas),
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WBallConstraint(c) => c.warmstart(mj_lambdas),
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WBallGroundConstraint(c) => c.warmstart(mj_lambdas),
            AnyJointVelocityConstraint::FixedConstraint(c) => c.warmstart(mj_lambdas),
            AnyJointVelocityConstraint::FixedGroundConstraint(c) => c.warmstart(mj_lambdas),
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WFixedConstraint(c) => c.warmstart(mj_lambdas),
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WFixedGroundConstraint(c) => c.warmstart(mj_lambdas),
            // AnyJointVelocityConstraint::GenericConstraint(c) => c.warmstart(mj_lambdas),
            // AnyJointVelocityConstraint::GenericGroundConstraint(c) => c.warmstart(mj_lambdas),
            // #[cfg(feature = "simd-is-enabled")]
            // AnyJointVelocityConstraint::WGenericConstraint(c) => c.warmstart(mj_lambdas),
            // #[cfg(feature = "simd-is-enabled")]
            // AnyJointVelocityConstraint::WGenericGroundConstraint(c) => c.warmstart(mj_lambdas),
            AnyJointVelocityConstraint::PrismaticConstraint(c) => c.warmstart(mj_lambdas),
            AnyJointVelocityConstraint::PrismaticGroundConstraint(c) => c.warmstart(mj_lambdas),
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WPrismaticConstraint(c) => c.warmstart(mj_lambdas),
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WPrismaticGroundConstraint(c) => c.warmstart(mj_lambdas),
            #[cfg(feature = "dim3")]
            AnyJointVelocityConstraint::RevoluteConstraint(c) => c.warmstart(mj_lambdas),
            #[cfg(feature = "dim3")]
            AnyJointVelocityConstraint::RevoluteGroundConstraint(c) => c.warmstart(mj_lambdas),
            #[cfg(feature = "dim3")]
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WRevoluteConstraint(c) => c.warmstart(mj_lambdas),
            #[cfg(feature = "dim3")]
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WRevoluteGroundConstraint(c) => c.warmstart(mj_lambdas),
            AnyJointVelocityConstraint::Empty => unreachable!(),
        }
    }

    pub fn solve(
        &mut self,
        jacobians: &DVector<Real>,
        mj_lambdas: &mut [DeltaVel<Real>],
        generic_mj_lambdas: &mut DVector<Real>,
    ) {
        match self {
            AnyJointVelocityConstraint::GenericConstraint3Dof(c) => {
                c.solve(jacobians, mj_lambdas, generic_mj_lambdas)
            }
            AnyJointVelocityConstraint::BallConstraint(c) => c.solve(mj_lambdas),
            AnyJointVelocityConstraint::BallGroundConstraint(c) => c.solve(mj_lambdas),
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WBallConstraint(c) => c.solve(mj_lambdas),
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WBallGroundConstraint(c) => c.solve(mj_lambdas),
            AnyJointVelocityConstraint::FixedConstraint(c) => c.solve(mj_lambdas),
            AnyJointVelocityConstraint::FixedGroundConstraint(c) => c.solve(mj_lambdas),
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WFixedConstraint(c) => c.solve(mj_lambdas),
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WFixedGroundConstraint(c) => c.solve(mj_lambdas),
            // AnyJointVelocityConstraint::GenericConstraint(c) => c.solve(mj_lambdas),
            // AnyJointVelocityConstraint::GenericGroundConstraint(c) => c.solve(mj_lambdas),
            // #[cfg(feature = "simd-is-enabled")]
            // AnyJointVelocityConstraint::WGenericConstraint(c) => c.solve(mj_lambdas),
            // #[cfg(feature = "simd-is-enabled")]
            // AnyJointVelocityConstraint::WGenericGroundConstraint(c) => c.solve(mj_lambdas),
            AnyJointVelocityConstraint::PrismaticConstraint(c) => c.solve(mj_lambdas),
            AnyJointVelocityConstraint::PrismaticGroundConstraint(c) => c.solve(mj_lambdas),
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WPrismaticConstraint(c) => c.solve(mj_lambdas),
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WPrismaticGroundConstraint(c) => c.solve(mj_lambdas),
            #[cfg(feature = "dim3")]
            AnyJointVelocityConstraint::RevoluteConstraint(c) => c.solve(mj_lambdas),
            #[cfg(feature = "dim3")]
            AnyJointVelocityConstraint::RevoluteGroundConstraint(c) => c.solve(mj_lambdas),
            #[cfg(feature = "dim3")]
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WRevoluteConstraint(c) => c.solve(mj_lambdas),
            #[cfg(feature = "dim3")]
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WRevoluteGroundConstraint(c) => c.solve(mj_lambdas),
            AnyJointVelocityConstraint::Empty => unreachable!(),
        }
    }

    pub fn writeback_impulses(&self, joints_all: &mut [JointGraphEdge]) {
        match self {
            AnyJointVelocityConstraint::GenericConstraint3Dof(c) => {
                c.writeback_impulses(joints_all)
            }
            AnyJointVelocityConstraint::BallConstraint(c) => c.writeback_impulses(joints_all),

            AnyJointVelocityConstraint::BallGroundConstraint(c) => c.writeback_impulses(joints_all),
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WBallConstraint(c) => c.writeback_impulses(joints_all),

            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WBallGroundConstraint(c) => {
                c.writeback_impulses(joints_all)
            }
            AnyJointVelocityConstraint::FixedConstraint(c) => c.writeback_impulses(joints_all),
            AnyJointVelocityConstraint::FixedGroundConstraint(c) => {
                c.writeback_impulses(joints_all)
            }
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WFixedConstraint(c) => c.writeback_impulses(joints_all),
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WFixedGroundConstraint(c) => {
                c.writeback_impulses(joints_all)
            }
            // AnyJointVelocityConstraint::GenericConstraint(c) => c.writeback_impulses(joints_all),
            // AnyJointVelocityConstraint::GenericGroundConstraint(c) => {
            //     c.writeback_impulses(joints_all)
            // }
            // #[cfg(feature = "simd-is-enabled")]
            // AnyJointVelocityConstraint::WGenericConstraint(c) => c.writeback_impulses(joints_all),
            // #[cfg(feature = "simd-is-enabled")]
            // AnyJointVelocityConstraint::WGenericGroundConstraint(c) => {
            //     c.writeback_impulses(joints_all)
            // }
            AnyJointVelocityConstraint::PrismaticConstraint(c) => c.writeback_impulses(joints_all),
            AnyJointVelocityConstraint::PrismaticGroundConstraint(c) => {
                c.writeback_impulses(joints_all)
            }
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WPrismaticConstraint(c) => c.writeback_impulses(joints_all),
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WPrismaticGroundConstraint(c) => {
                c.writeback_impulses(joints_all)
            }
            #[cfg(feature = "dim3")]
            AnyJointVelocityConstraint::RevoluteConstraint(c) => c.writeback_impulses(joints_all),
            #[cfg(feature = "dim3")]
            AnyJointVelocityConstraint::RevoluteGroundConstraint(c) => {
                c.writeback_impulses(joints_all)
            }
            #[cfg(feature = "dim3")]
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WRevoluteConstraint(c) => c.writeback_impulses(joints_all),
            #[cfg(feature = "dim3")]
            #[cfg(feature = "simd-is-enabled")]
            AnyJointVelocityConstraint::WRevoluteGroundConstraint(c) => {
                c.writeback_impulses(joints_all)
            }
            AnyJointVelocityConstraint::Empty => unreachable!(),
        }
    }
}
