use na::DVectorSliceMut;

use crate::dynamics::{Articulation, IntegrationParameters, RigidBodyVelocity};
use crate::math::{Isometry, JacobianSliceMut, Real, Vector, SPATIAL_DIM};

/// A articulation that allows all the relative degrees of freedom between two multibody links.
///
/// This articulation can only be added between a `Ground` body (as parent) and any other body.
#[derive(Copy, Clone, Debug)]
pub struct FreeArticulation {
    position: Isometry<Real>,
}

impl FreeArticulation {
    /// Creates a free articulation with the given initial position of the descendent, relative to the ground.
    pub fn new(position: Isometry<Real>) -> Self {
        FreeArticulation { position }
    }

    fn apply_displacement(&mut self, disp: &RigidBodyVelocity) {
        let disp = Isometry::new(disp.linvel, disp.angvel);
        self.position = Isometry::from_parts(
            disp.translation * self.position.translation,
            disp.rotation * self.position.rotation,
        )
    }
}

impl Articulation for FreeArticulation {
    fn ndofs(&self) -> usize {
        SPATIAL_DIM
    }

    fn body_to_parent(&self, _: &Vector<Real>, _: &Vector<Real>) -> Isometry<Real> {
        self.position
    }

    fn update_jacobians(&mut self, _: &Vector<Real>, _: &[Real]) {}

    fn jacobian(&self, _: &Isometry<Real>, out: &mut JacobianSliceMut<Real>) {
        out.fill_diagonal(1.0);
    }

    fn jacobian_dot(&self, _: &Isometry<Real>, _: &mut JacobianSliceMut<Real>) {}

    fn jacobian_dot_veldiff_mul_coordinates(
        &self,
        _: &Isometry<Real>,
        _: &[Real],
        _: &mut JacobianSliceMut<Real>,
    ) {
    }

    fn integrate(&mut self, dt: Real, vels: &[Real]) {
        let disp = RigidBodyVelocity::from_slice(vels) * dt;
        self.apply_displacement(&disp);
    }

    fn apply_displacement(&mut self, disp: &[Real]) {
        let disp = RigidBodyVelocity::from_slice(disp);
        self.apply_displacement(&disp);
    }

    fn jacobian_mul_coordinates(&self, vels: &[Real]) -> RigidBodyVelocity {
        RigidBodyVelocity::from_slice(vels)
    }

    fn jacobian_dot_mul_coordinates(&self, _: &[Real]) -> RigidBodyVelocity {
        // The jacobian derivative is zero.
        RigidBodyVelocity::zero()
    }

    fn default_damping(&self, _: &mut DVectorSliceMut<Real>) {}

    #[inline]
    fn clone(&self) -> Box<dyn Articulation> {
        Box::new(*self)
    }
}
