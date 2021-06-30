use na::{DVectorSliceMut, Isometry3, Unit, Vector3};

use crate::dynamics::{
    Articulation, IntegrationParameters, PrismaticArticulation, RevoluteArticulation,
    RigidBodyVelocity,
};
use crate::math::{JacobianSliceMut, Real};
use approx::AbsDiffEq;

/// A articulation that allows 1 rotational and 2 translational degrees of freedom.
#[derive(Copy, Clone, Debug)]
pub struct PlanarArticulation {
    prism1: PrismaticArticulation,
    prism2: PrismaticArticulation,
    revo: RevoluteArticulation,
}

impl PlanarArticulation {
    /// Create a new planar articulation where both translational degrees of freedoms are along the provide axes.
    ///
    /// The rotational degree of freedom is along an axis orthogonal to `axis1` and `axis2`. Ideally, the two
    /// provided axes should be orthogonal. All axis are in the local coordinate space of the attached multibody links.
    ///
    /// Panics if `axis1` and `axis2` are near-collinear.
    pub fn new(
        axis1: Unit<Vector3<Real>>,
        axis2: Unit<Vector3<Real>>,
        pos1: Real,
        pos2: Real,
        angle: Real,
    ) -> Self {
        let cross = axis1.cross(&*axis2);
        let normal = Unit::try_new(cross, Real::default_epsilon())
            .expect("A planar articulation cannot be defined from two collinear axis.");
        let prism1 = PrismaticArticulation::new(axis1, pos1);
        let prism2 = PrismaticArticulation::new(axis2, pos2);
        let revo = RevoluteArticulation::new(normal, angle);

        PlanarArticulation {
            prism1,
            prism2,
            revo,
        }
    }
}

impl Articulation for PlanarArticulation {
    #[inline]
    fn ndofs(&self) -> usize {
        3
    }

    fn body_to_parent(
        &self,
        parent_shift: &Vector3<Real>,
        body_shift: &Vector3<Real>,
    ) -> Isometry3<Real> {
        self.prism1.translation()
            * self.prism2.translation()
            * self.revo.body_to_parent(parent_shift, body_shift)
    }

    fn update_jacobians(&mut self, body_shift: &Vector3<Real>, vels: &[Real]) {
        self.prism1.update_jacobians(body_shift, vels);
        self.prism2.update_jacobians(body_shift, &vels[1..]);
        self.revo.update_jacobians(body_shift, &vels[2..]);
    }

    fn jacobian(&self, transform: &Isometry3<Real>, out: &mut JacobianSliceMut<Real>) {
        self.prism1.jacobian(transform, &mut out.columns_mut(0, 1));
        self.prism2.jacobian(transform, &mut out.columns_mut(1, 1));
        self.revo.jacobian(transform, &mut out.columns_mut(2, 1));
    }

    fn jacobian_dot(&self, transform: &Isometry3<Real>, out: &mut JacobianSliceMut<Real>) {
        self.prism1
            .jacobian_dot(transform, &mut out.columns_mut(0, 1));
        self.prism2
            .jacobian_dot(transform, &mut out.columns_mut(1, 1));
        self.revo
            .jacobian_dot(transform, &mut out.columns_mut(2, 1));
    }

    fn jacobian_dot_veldiff_mul_coordinates(
        &self,
        transform: &Isometry3<Real>,
        vels: &[Real],
        out: &mut JacobianSliceMut<Real>,
    ) {
        self.prism1.jacobian_dot_veldiff_mul_coordinates(
            transform,
            vels,
            &mut out.columns_mut(0, 1),
        );
        self.prism2.jacobian_dot_veldiff_mul_coordinates(
            transform,
            &[vels[1]],
            &mut out.columns_mut(1, 1),
        );
        self.revo.jacobian_dot_veldiff_mul_coordinates(
            transform,
            &[vels[2]],
            &mut out.columns_mut(2, 1),
        );
    }

    fn jacobian_mul_coordinates(&self, vels: &[Real]) -> RigidBodyVelocity {
        self.prism1.jacobian_mul_coordinates(vels)
            + self.prism2.jacobian_mul_coordinates(&[vels[1]])
            + self.revo.jacobian_mul_coordinates(&[vels[2]])
    }

    fn jacobian_dot_mul_coordinates(&self, vels: &[Real]) -> RigidBodyVelocity {
        // NOTE: The two folowing are zero.
        // self.prism1.jacobian_dot_mul_coordinates(vels)       +
        // self.prism2.jacobian_dot_mul_coordinates(&[vels[1]]) +
        self.revo.jacobian_dot_mul_coordinates(&[vels[2]])
    }

    fn default_damping(&self, out: &mut DVectorSliceMut<Real>) {
        self.prism1.default_damping(&mut out.rows_mut(0, 1));
        self.prism2.default_damping(&mut out.rows_mut(1, 1));
        self.revo.default_damping(&mut out.rows_mut(2, 1));
    }

    fn integrate(&mut self, dt: Real, vels: &[Real]) {
        self.prism1.integrate(dt, vels);
        self.prism2.integrate(dt, &[vels[1]]);
        self.revo.integrate(dt, &[vels[2]]);
    }

    fn apply_displacement(&mut self, disp: &[Real]) {
        self.prism1.apply_displacement(disp);
        self.prism2.apply_displacement(&[disp[1]]);
        self.revo.apply_displacement(&[disp[2]]);
    }

    #[inline]
    fn clone(&self) -> Box<dyn Articulation> {
        Box::new(*self)
    }

    /*
    fn num_velocity_constraints(&self) -> usize {
        self.prism1.num_velocity_constraints()
            + self.prism2.num_velocity_constraints()
            + self.revo.num_velocity_constraints()
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
        self.revo.velocity_constraints(
            parameters,
            multibody,
            link,
            assembly_id,
            dof_id + 2,
            ext_vels,
            ground_j_id,
            jacobians,
            constraints,
        );
    }

    fn num_position_constraints(&self) -> usize {
        // NOTE: we don't test if constraints exist to simplify indexing.
        3
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
        } else if i == 1 {
            self.prism2
                .position_constraint(0, multibody, link, handle, dof_id + 1, jacobians)
        } else {
            self.revo
                .position_constraint(0, multibody, link, handle, dof_id + 2, jacobians)
        }
    }

     */
}

prismatic_motor_limit_methods_1!(PlanarArticulation, prism1);
prismatic_motor_limit_methods_2!(PlanarArticulation, prism2);
revolute_motor_limit_methods!(PlanarArticulation, revo);
