use na::{self, DVectorSliceMut};

use crate::dynamics::{Articulation, IntegrationParameters, RigidBodyVelocity};
use crate::math::{Isometry, JacobianSliceMut, Real, Translation, Vector, DIM};

/// A articulation that allows only all the translational degrees of freedom between two multibody links.
#[derive(Copy, Clone, Debug)]
pub struct CartesianArticulation {
    position: Vector<Real>,
}

impl CartesianArticulation {
    /// Create a cartesian articulation with an initial position given by `position`.
    pub fn new(position: Vector<Real>) -> Self {
        CartesianArticulation { position }
    }
}

impl Articulation for CartesianArticulation {
    #[inline]
    fn ndofs(&self) -> usize {
        DIM
    }

    fn body_to_parent(
        &self,
        parent_shift: &Vector<Real>,
        body_shift: &Vector<Real>,
    ) -> Isometry<Real> {
        let t = Translation::from(parent_shift - body_shift + self.position);
        Isometry::from_parts(t, na::one())
    }

    fn update_jacobians(&mut self, _: &Vector<Real>, _: &[Real]) {}

    fn jacobian(&self, _: &Isometry<Real>, out: &mut JacobianSliceMut<Real>) {
        out.fill_diagonal(1.0)
    }

    fn jacobian_dot(&self, _: &Isometry<Real>, _: &mut JacobianSliceMut<Real>) {}

    fn jacobian_dot_veldiff_mul_coordinates(
        &self,
        _: &Isometry<Real>,
        _: &[Real],
        _: &mut JacobianSliceMut<Real>,
    ) {
    }

    fn jacobian_mul_coordinates(&self, vels: &[Real]) -> RigidBodyVelocity {
        RigidBodyVelocity::new(Vector::from_row_slice(&vels[..DIM]), na::zero())
    }

    fn jacobian_dot_mul_coordinates(&self, _: &[Real]) -> RigidBodyVelocity {
        RigidBodyVelocity::zero()
    }

    fn default_damping(&self, _: &mut DVectorSliceMut<Real>) {}

    fn integrate(&mut self, dt: Real, vels: &[Real]) {
        self.position += Vector::from_row_slice(&vels[..DIM]) * dt;
    }

    fn apply_displacement(&mut self, disp: &[Real]) {
        self.position += Vector::from_row_slice(&disp[..DIM]);
    }

    #[inline]
    fn clone(&self) -> Box<dyn Articulation> {
        Box::new(*self)
    }
}
