#![macro_use]

use na::{self, DVectorSliceMut, Unit};

use crate::dynamics::{
    Articulation, ArticulationMotor, IntegrationParameters, RigidBodyVelocity, UnitArticulation,
};
use crate::math::{Isometry, JacobianSliceMut, Real, Rotation, Translation, Vector, DIM};

/// A unit articulation that allows only one translational degree on freedom.
#[derive(Copy, Clone, Debug)]
pub struct PrismaticArticulation {
    axis: Unit<Vector<Real>>,
    jacobian: RigidBodyVelocity,

    offset: Real,

    min_offset: Option<Real>,
    max_offset: Option<Real>,
    motor: ArticulationMotor<Real>,
}

impl PrismaticArticulation {
    /// Create a new prismatic articulation where the allowed traslation is defined along the provided axis.
    ///
    /// The axis is expressed in the local coordinate system of the two multibody links attached to this articulation.
    #[cfg(feature = "dim2")]
    pub fn new(axis: Unit<Vector<Real>>, offset: Real) -> Self {
        PrismaticArticulation {
            axis,
            jacobian: Velocity::zero(),
            offset,
            min_offset: None,
            max_offset: None,
            motor: ArticulationMotor::new(),
        }
    }

    /// Create a new prismatic articulation where the allowed traslation is defined along the provided axis.
    ///
    /// The axis is expressed in the local coordinate system of the two multibody links attached to this articulation.
    #[cfg(feature = "dim3")]
    pub fn new(axis: Unit<Vector<Real>>, offset: Real) -> Self {
        PrismaticArticulation {
            axis,
            jacobian: RigidBodyVelocity::zero(),
            offset,
            min_offset: None,
            max_offset: None,
            motor: ArticulationMotor::new(),
        }
    }

    /// The relative displacement of the attached multibody links along the articulation axis.
    pub fn offset(&self) -> Real {
        self.offset
    }

    /// The relative translation of the attached multibody links along the articulation axis.
    pub fn translation(&self) -> Translation<Real> {
        Translation::from(*self.axis * self.offset)
    }

    /// The lower limit of the relative displacement of the attached multibody links along the articulation axis.
    pub fn min_offset(&self) -> Option<Real> {
        self.min_offset
    }

    /// The upper limit of the relative displacement of the attached multibody links along the articulation axis.
    pub fn max_offset(&self) -> Option<Real> {
        self.max_offset
    }

    /// Disable the lower limit of the relative displacement of the attached multibody links along the articulation axis.
    pub fn disable_min_offset(&mut self) {
        self.min_offset = None;
    }

    /// Disable the upper limit of the relative displacement of the attached multibody links along the articulation axis.
    pub fn disable_max_offset(&mut self) {
        self.max_offset = None;
    }

    /// Set the lower limit of the relative displacement of the attached multibody links along the articulation axis.
    pub fn enable_min_offset(&mut self, limit: Real) {
        self.min_offset = Some(limit);
        self.assert_limits();
    }

    /// Set the upper limit of the relative displacement of the attached multibody links along the articulation axis.
    pub fn enable_max_offset(&mut self, limit: Real) {
        self.max_offset = Some(limit);
        self.assert_limits();
    }

    /// Returns `true` if the articulation motor is enabled.
    pub fn is_linear_motor_enabled(&self) -> bool {
        self.motor.enabled
    }

    /// Enable the articulation motor.
    pub fn enable_linear_motor(&mut self) {
        self.motor.enabled = true
    }

    /// Disable the articulation motor.
    pub fn disable_linear_motor(&mut self) {
        self.motor.enabled = false;
    }

    /// The desired relative velocity to be enforced by the articulation motor.
    pub fn desired_linear_motor_velocity(&self) -> Real {
        self.motor.desired_velocity
    }

    /// Set the desired relative velocity to be enforced by the articulation motor.
    pub fn set_desired_linear_motor_velocity(&mut self, vel: Real) {
        self.motor.desired_velocity = vel;
    }

    /// The max linear velocity that the articulation motor will attempt.
    pub fn max_linear_motor_velocity(&self) -> Real {
        self.motor.max_velocity
    }

    /// Set the maximum linear velocity that the articulation motor will attempt.
    pub fn set_max_linear_motor_velocity(&mut self, max_vel: Real) {
        self.motor.max_velocity = max_vel;
    }

    /// The maximum force that can be output by the articulation motor.
    pub fn max_linear_motor_force(&self) -> Real {
        self.motor.max_force
    }

    /// Set the maximum force that can be output by the articulation motor.
    pub fn set_max_linear_motor_force(&mut self, force: Real) {
        self.motor.max_force = force;
    }

    fn assert_limits(&self) {
        if let (Some(min_offset), Some(max_offset)) = (self.min_offset, self.max_offset) {
            assert!(
                min_offset <= max_offset,
                "PrismaticArticulation articulation limits: the min offset must be larger than (or equal to) the max offset.");
        }
    }
}

impl Articulation for PrismaticArticulation {
    #[inline]
    fn ndofs(&self) -> usize {
        1
    }

    #[cfg(feature = "dim3")]
    fn body_to_parent(
        &self,
        parent_shift: &Vector<Real>,
        body_shift: &Vector<Real>,
    ) -> Isometry<Real> {
        let trans = Translation::from(parent_shift - body_shift + self.axis.as_ref() * self.offset);
        Isometry::from_parts(trans, Rotation::identity())
    }

    #[cfg(feature = "dim2")]
    fn body_to_parent(
        &self,
        parent_shift: &Vector<Real>,
        body_shift: &Vector<Real>,
    ) -> Isometry<Real> {
        let trans = Translation::from(parent_shift - body_shift + self.axis.as_ref() * self.offset);
        Isometry::from_parts(trans, Rotation::identity())
    }

    fn update_jacobians(&mut self, _: &Vector<Real>, _: &[Real]) {}

    fn jacobian(&self, transform: &Isometry<Real>, out: &mut JacobianSliceMut<Real>) {
        let transformed_axis = transform * self.axis;
        out.fixed_rows_mut::<DIM>(0)
            .copy_from(transformed_axis.as_ref())
    }

    fn jacobian_dot(&self, _: &Isometry<Real>, _: &mut JacobianSliceMut<Real>) {}

    fn jacobian_dot_veldiff_mul_coordinates(
        &self,
        _: &Isometry<Real>,
        _: &[Real],
        _: &mut JacobianSliceMut<Real>,
    ) {
    }

    fn default_damping(&self, _: &mut DVectorSliceMut<Real>) {}

    fn integrate(&mut self, dt: Real, vels: &[Real]) {
        self.offset += vels[0] * dt
    }

    fn apply_displacement(&mut self, disp: &[Real]) {
        self.offset += disp[0]
    }

    fn jacobian_mul_coordinates(&self, acc: &[Real]) -> RigidBodyVelocity {
        RigidBodyVelocity::new(self.axis.as_ref() * acc[0], na::zero())
    }

    fn jacobian_dot_mul_coordinates(&self, _: &[Real]) -> RigidBodyVelocity {
        RigidBodyVelocity::zero()
    }

    #[inline]
    fn clone(&self) -> Box<dyn Articulation> {
        Box::new(*self)
    }

    /*
    fn num_velocity_constraints(&self) -> usize {
        articulation::unit_articulation_num_velocity_constraints(self)
    }

    fn velocity_constraints(
        &self,
        parameters: &IntegrationParameters,
        multibody: &Multibody,
        link: &MultibodyLink,
        assembly_id: usize,
        dof_id: usize,
        ext_vels: &[Real],
        ground_j_id: &mut usize,
        jacobians: &mut [Real],
        constraints: &mut ConstraintSet<Real, (), (), usize>,
    ) {
        articulation::unit_articulation_velocity_constraints(
            self,
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
    }

    fn num_position_constraints(&self) -> usize {
        if self.min_offset.is_some() || self.max_offset.is_some() {
            1
        } else {
            0
        }
    }

    fn position_constraint(
        &self,
        _: usize,
        multibody: &Multibody,
        link: &MultibodyLink,
        handle: BodyPartHandle<()>,
        dof_id: usize,
        jacobians: &mut [Real],
    ) -> Option<GenericNonlinearConstraint<Real, ()>> {
        articulation::unit_articulation_position_constraint(
            self, multibody, link, handle, dof_id, false, jacobians,
        )
    }
     */
}

impl UnitArticulation for PrismaticArticulation {
    fn position(&self) -> Real {
        self.offset
    }

    fn motor(&self) -> &ArticulationMotor<Real> {
        &self.motor
    }

    fn min_position(&self) -> Option<Real> {
        self.min_offset
    }

    fn max_position(&self) -> Option<Real> {
        self.max_offset
    }
}

#[cfg(feature = "dim3")]
macro_rules! prismatic_motor_limit_methods (
    ($ty: ident, $prism: ident) => {
        _prismatic_motor_limit_methods!(
            $ty,
            $prism,
            min_offset,
            max_offset,
            disable_min_offset,
            disable_max_offset,
            enable_min_offset,
            enable_max_offset,
            is_linear_motor_enabled,
            enable_linear_motor,
            disable_linear_motor,
            desired_linear_motor_velocity,
            set_desired_linear_motor_velocity,
            max_linear_motor_force,
            set_max_linear_motor_force);
    }
);

#[cfg(feature = "dim3")]
macro_rules! prismatic_motor_limit_methods_1 (
    ($ty: ident, $prism: ident) => {
        _prismatic_motor_limit_methods!(
            $ty,
            $prism,
            min_offset_1,
            max_offset_1,
            disable_min_offset_1,
            disable_max_offset_1,
            enable_min_offset_1,
            enable_max_offset_1,
            is_linear_motor_enabled_1,
            enable_linear_motor_1,
            disable_linear_motor_1,
            desired_linear_motor_velocity_1,
            set_desired_linear_motor_velocity_1,
            max_linear_motor_force_1,
            set_max_linear_motor_force_1);
    }
);

#[cfg(feature = "dim3")]
macro_rules! prismatic_motor_limit_methods_2 (
    ($ty: ident, $prism: ident) => {
        _prismatic_motor_limit_methods!(
            $ty,
            $prism,
            min_offset_2,
            max_offset_2,
            disable_min_offset_2,
            disable_max_offset_2,
            enable_min_offset_2,
            enable_max_offset_2,
            is_linear_motor_enabled_2,
            enable_linear_motor_2,
            disable_linear_motor_2,
            desired_linear_motor_velocity_2,
            set_desired_linear_motor_velocity_2,
            max_linear_motor_force2,
            set_max_linear_motor_force_2);
    }
);

#[cfg(feature = "dim3")]
macro_rules! _prismatic_motor_limit_methods (
    ($ty: ident, $prism: ident,
     $min_offset:         ident,
     $max_offset:         ident,
     $disable_min_offset: ident,
     $disable_max_offset: ident,
     $enable_min_offset:  ident,
     $enable_max_offset:  ident,
     $is_motor_enabled:  ident,
     $enable_motor:      ident,
     $disable_motor:     ident,
     $desired_motor_velocity:     ident,
     $set_desired_motor_velocity: ident,
     $max_motor_force:           ident,
     $set_max_motor_force:       ident
     ) => {
        impl $ty {
            /// The lower limit of the relative translational displacement of the attached multibody links along the articulation axis.
            pub fn $min_offset(&self) -> Option<Real> {
                self.$prism.min_offset()
            }

            /// The upper limit of the relative translational displacement of the attached multibody links along the articulation axis.
            pub fn $max_offset(&self) -> Option<Real> {
                self.$prism.max_offset()
            }

            /// Disable the lower limit of the relative translational displacement of the attached multibody links along the articulation axis.
            pub fn $disable_min_offset(&mut self) {
                self.$prism.disable_max_offset();
            }

            /// Disable the upper limit of the relative translational displacement of the attached multibody links along the articulation axis.
            pub fn $disable_max_offset(&mut self) {
                self.$prism.disable_max_offset();
            }

            /// Set the lower limit of the relative translational displacement of the attached multibody links along the articulation axis.
            pub fn $enable_min_offset(&mut self, limit: Real) {
                self.$prism.enable_min_offset(limit);
            }

            /// Set the upper limit of the relative translational displacement of the attached multibody links along the articulation axis.
            pub fn $enable_max_offset(&mut self, limit: Real) {
                self.$prism.enable_max_offset(limit)
            }

            /// Returns `true` if the articulation translational motor is enabled.
            pub fn $is_motor_enabled(&self) -> bool {
                self.$prism.is_linear_motor_enabled()
            }

            /// Enable the articulation translational motor.
            pub fn $enable_motor(&mut self) {
                self.$prism.enable_linear_motor()
            }

            /// Disable the articulation translational motor.
            pub fn $disable_motor(&mut self) {
                self.$prism.disable_linear_motor()
            }

            /// The desired relative translational velocity to be enforced by the articulation motor.
            pub fn $desired_motor_velocity(&self) -> Real {
                self.$prism.desired_linear_motor_velocity()
            }

            /// Set the desired relative translational velocity to be enforced by the articulation motor.
            pub fn $set_desired_motor_velocity(&mut self, vel: Real) {
                self.$prism.set_desired_linear_motor_velocity(vel)
            }

            /// The maximum force that can be output by the articulation translational motor.
            pub fn $max_motor_force(&self) -> Real {
                self.$prism.max_linear_motor_force()
            }

            /// Set the maximum force that can be output by the articulation translational motor.
            pub fn $set_max_motor_force(&mut self, force: Real) {
                self.$prism.set_max_linear_motor_force(force)
            }
        }
    }
);
