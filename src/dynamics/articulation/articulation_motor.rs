use crate::math::Real;
use num::Zero;

/// Description of a motor applied to a joint.
#[derive(Copy, Clone, Debug)]
pub struct ArticulationMotor<V> {
    /// The velocity the motor will attempt to reach.
    pub desired_velocity: V,
    /// The maximum velocity the motor will attempt to reach.
    pub max_velocity: Real,
    /// The maximum force deliverable by the motor.
    pub max_force: Real,
    /// Whether or not the motor is active.
    pub enabled: bool,
}

impl<V: Zero> ArticulationMotor<V> {
    /// Create a disable motor with zero desired velocity.
    ///
    /// The max force is initialized to a virtually infinite value, i.e., `N::max_value()`.
    pub fn new() -> Self {
        ArticulationMotor {
            desired_velocity: V::zero(),
            max_velocity: Real::MAX,
            max_force: Real::MAX,
            enabled: false,
        }
    }

    /// The limits of the impulse applicable by the motor on the body parts.
    pub fn impulse_limits(&self) -> (Real, Real) {
        (-self.max_force, self.max_force)
    }
}

impl<V: Zero> Default for ArticulationMotor<V> {
    fn default() -> Self {
        Self::new()
    }
}
