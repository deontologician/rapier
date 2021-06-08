use super::multibody_link::{MultibodyLink, MultibodyLinkVec};
use super::multibody_workspace::MultibodyWorkspace;
use crate::data::{BundleSet, ComponentSet, ComponentSetMut};
use crate::dynamics::{
    Articulation, IntegrationParameters, RigidBodyMassProps, RigidBodyPosition, RigidBodyType,
    RigidBodyVelocity,
};
use crate::math::{
    AngDim, AngVector, AngularInertia, Dim, Isometry, Jacobian, Matrix, Point, Real, Vector, DIM,
    SPATIAL_DIM,
};
use crate::prelude::RigidBodyForces;
use crate::utils::{IndexMut2, WAngularInertia, WCross, WCrossMatrix};
use na::{
    self, DMatrix, DVector, DVectorSlice, DVectorSliceMut, Dynamic, OMatrix, SMatrix, SVector, LU,
};

#[cfg(feature = "dim2")]
const ANG_DIM: usize = 2;
#[cfg(feature = "dim3")]
const ANG_DIM: usize = 3;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Force {
    linear: Vector<Real>,
    angular: AngVector<Real>,
}

impl Force {
    fn new(linear: Vector<Real>, angular: AngVector<Real>) -> Self {
        Self { linear, angular }
    }

    fn as_vector(&self) -> &SVector<Real, SPATIAL_DIM> {
        unsafe { std::mem::transmute(self) }
    }
}

#[cfg(feature = "dim2")]
fn concat_rb_mass_matrix(mass: Real, inertia: Real) -> SMatrix<Real, SPATIAL_DIM, SPATIAL_DIM> {
    let mut result = SMatrix::<Real, SPATIAL_DIM, SPATIAL_DIM>::zeros();
    result[(0, 0)] = mass;
    result[(1, 1)] = mass;
    result[(2, 2)] = inertia;
    result
}

#[cfg(feature = "dim3")]
fn concat_rb_mass_matrix(
    mass: Real,
    inertia: Matrix<Real>,
) -> SMatrix<Real, SPATIAL_DIM, SPATIAL_DIM> {
    let mut result = SMatrix::<Real, SPATIAL_DIM, SPATIAL_DIM>::zeros();
    result[(0, 0)] = mass;
    result[(1, 1)] = mass;
    result[(2, 2)] = mass;
    result
        .fixed_slice_mut::<ANG_DIM, ANG_DIM>(DIM, DIM)
        .copy_from(&inertia);
    result
}

/// An articulated body simulated using the reduced-coordinates approach.
pub struct Multibody {
    links: MultibodyLinkVec,
    velocities: DVector<Real>,
    damping: DVector<Real>,
    accelerations: DVector<Real>,
    impulses: DVector<Real>,
    body_jacobians: Vec<Jacobian<Real>>,
    // FIXME: use sparse matrices.
    augmented_mass: DMatrix<Real>,
    inv_augmented_mass: LU<Real, Dynamic, Dynamic>,
    ndofs: usize,
    companion_id: usize,

    /*
     * Workspaces.
     */
    workspace: MultibodyWorkspace,
    coriolis_v: Vec<OMatrix<Real, Dim, Dynamic>>,
    coriolis_w: Vec<OMatrix<Real, AngDim, Dynamic>>,
    i_coriolis_dt: Jacobian<Real>,
    /*
     * Constraint resolution data.
     * FIXME: we should void explicitly generating those constraints by
     * just iterating on all joints at each step of the resolution.
     */
    // solver_workspace: Option<SolverWorkspace<Real, (), ()>>,
}

impl Multibody {
    /// Creates a new multibody with no link.
    fn new() -> Self {
        Multibody {
            links: MultibodyLinkVec(Vec::new()),
            velocities: DVector::zeros(0),
            damping: DVector::zeros(0),
            accelerations: DVector::zeros(0),
            impulses: DVector::zeros(0),
            body_jacobians: Vec::new(),
            augmented_mass: DMatrix::zeros(0, 0),
            inv_augmented_mass: LU::new(DMatrix::zeros(0, 0)),
            ndofs: 0,
            companion_id: 0,
            workspace: MultibodyWorkspace::new(),
            coriolis_v: Vec::new(),
            coriolis_w: Vec::new(),
            i_coriolis_dt: Jacobian::zeros(0),
            // solver_workspace: Some(SolverWorkspace::new()),
        }
    }

    /// The first link of this multibody.
    #[inline]
    pub fn root(&self) -> &MultibodyLink {
        &self.links[0]
    }

    /// Mutable reference to the first link of this multibody.
    #[inline]
    pub fn root_mut(&mut self) -> &mut MultibodyLink {
        &mut self.links[0]
    }

    /// Reference `i`-th multibody link of this multibody.
    ///
    /// Return `None` if there is less than `i + 1` multibody links.
    #[inline]
    pub fn link(&self, id: usize) -> Option<&MultibodyLink> {
        self.links.get(id)
    }

    /// Mutable reference to the multibody link with the given id.
    ///
    /// Return `None` if the given id does not identifies a multibody link part of `self`.
    #[inline]
    pub fn link_mut(&mut self, id: usize) -> Option<&mut MultibodyLink> {
        self.links.get_mut(id)
    }

    /// The links of this multibody with the given `name`.
    pub fn links_with_name<'a>(
        &'a self,
        name: &'a str,
    ) -> impl Iterator<Item = (usize, &'a MultibodyLink)> {
        self.links
            .iter()
            .enumerate()
            .filter(move |(_i, l)| l.name == name)
    }

    /// The number of links on this multibody.
    pub fn num_links(&self) -> usize {
        self.links.len()
    }

    /// Iterator through all the links of this multibody.
    ///
    /// All link are guaranteed to be yielded before its descendant.
    pub fn links(&self) -> impl Iterator<Item = &MultibodyLink> {
        self.links.iter()
    }

    /// Mutable iterator through all the links of this multibody.
    ///
    /// All link are guaranteed to be yielded before its descendant.
    pub fn links_mut(&mut self) -> impl Iterator<Item = &mut MultibodyLink> {
        self.links.iter_mut()
    }

    /// The vector of damping applied to this multibody.
    #[inline]
    pub fn damping(&self) -> &DVector<Real> {
        &self.damping
    }

    /// Mutable vector of damping applied to this multibody.
    #[inline]
    pub fn damping_mut(&mut self) -> &mut DVector<Real> {
        &mut self.damping
    }

    /*
    fn add_link(
        &mut self,
        parent: Option<usize>,
        mut dof: Box<dyn Articulation>,
        parent_shift: Vector<Real>,
        body_shift: Vector<Real>,
        local_inertia: Inertia<Real>,
        local_com: Point<Real>,
    ) -> &mut MultibodyLink {
        assert!(
            parent.is_none() || !self.links.is_empty(),
            "Multibody::build_body: invalid parent id."
        );

        /*
         * Compute the indices.
         */
        let assembly_id = self.velocities.len();
        let impulse_id = self.impulses.len();
        let internal_id = self.links.len();

        /*
         * Grow the buffers.
         */
        let ndofs = dof.ndofs();
        let nimpulses = dof.nimpulses();
        self.grow_buffers(ndofs, nimpulses);
        self.ndofs += ndofs;

        /*
         * Setup default damping.
         */
        dof.default_damping(&mut self.damping.rows_mut(assembly_id, ndofs));

        /*
         * Create the multibody.
         */
        dof.update_jacobians(&body_shift, &self.velocities.as_slice()[assembly_id..]);
        let local_to_parent = dof.body_to_parent(&parent_shift, &body_shift);
        let local_to_world;
        let parent_to_world;

        let parent_internal_id;
        if let Some(parent) = parent {
            parent_internal_id = parent;
            let parent_link = &mut self.links[parent_internal_id];
            parent_link.is_leaf = false;
            parent_to_world = parent_link.local_to_world;
            local_to_world = parent_link.local_to_world * local_to_parent;
        } else {
            parent_internal_id = 0;
            parent_to_world = Isometry::identity();
            local_to_world = local_to_parent;
        }

        let rb = MultibodyLink::new(
            internal_id,
            assembly_id,
            impulse_id,
            parent_internal_id,
            dof,
            parent_shift,
            body_shift,
            parent_to_world,
            local_to_world,
            local_to_parent,
            local_inertia,
            local_com,
        );

        self.links.push(rb);
        self.workspace.resize(self.links.len(), self.ndofs);

        &mut self.links[internal_id]
    }

    fn grow_buffers(&mut self, ndofs: usize, nimpulses: usize) {
        let len = self.velocities.len();
        self.velocities.resize_vertically_mut(len + ndofs, 0.0);
        self.damping.resize_vertically_mut(len + ndofs, 0.0);
        self.accelerations.resize_vertically_mut(len + ndofs, 0.0);
        self.body_jacobians.push(Jacobian::zeros(0));

        let len = self.impulses.len();
        self.impulses.resize_vertically_mut(len + nimpulses, 0.0);
    }
     */

    fn update_acceleration<Bodies>(&mut self, bodies: &Bodies)
    where
        Bodies: ComponentSet<RigidBodyMassProps>
            + ComponentSet<RigidBodyForces>
            + ComponentSet<RigidBodyVelocity>,
    {
        self.accelerations.fill(0.0);

        for i in 0..self.links.len() {
            let link = &self.links[i];

            let (rb_vels, rb_mprops, rb_forces): (
                &RigidBodyVelocity,
                &RigidBodyMassProps,
                &RigidBodyForces,
            ) = bodies.index_bundle(link.rigid_body.0);

            let mut acc = link.velocity_dot_wrt_joint;

            if i != 0 {
                let parent_id = link.parent_internal_id;
                let parent_link = &self.links[parent_id];

                let (parent_rb_vels, parent_rb_mprops): (&RigidBodyVelocity, &RigidBodyMassProps) =
                    bodies.index_bundle(parent_link.rigid_body.0);

                acc += self.workspace.accs[parent_id];
                acc.linvel += parent_rb_vels.angvel.gcross(link.velocity_wrt_joint.linvel);
                #[cfg(feature = "dim3")]
                {
                    acc.angvel += parent_rb_vels.angvel.cross(&link.velocity_wrt_joint.angvel);
                }

                let shift = rb_mprops.world_com - parent_rb_mprops.world_com;
                let dvel = rb_vels.linvel - parent_rb_vels.linvel;

                acc.linvel += parent_rb_vels.angvel.gcross(dvel);
                acc.linvel += self.workspace.accs[parent_id].angvel.gcross(shift);
            }

            self.workspace.accs[i] = acc;

            // TODO: should gyroscopic forces already be computed by the rigid-body itself
            //       (at the same time that we add the gravity force)?
            let gyroscopic;
            let rb_inertia = rb_mprops.effective_angular_inertia();
            let rb_mass = rb_mprops.effective_mass();

            #[cfg(feature = "dim3")]
            {
                gyroscopic = rb_vels.angvel.cross(&(rb_inertia * rb_vels.angvel));
            }
            #[cfg(feature = "dim2")]
            {
                gyroscopic = 0.0;
            }

            let external_forces = Force::new(
                rb_forces.force + rb_mass * acc.linvel,
                rb_forces.torque - gyroscopic - rb_inertia * acc.angvel,
            );
            self.accelerations.gemv_tr(
                1.0,
                &self.body_jacobians[i],
                external_forces.as_vector(),
                1.0,
            );
        }

        self.accelerations
            .cmpy(-1.0, &self.damping, &self.velocities, 1.0);

        assert!(self.inv_augmented_mass.solve_mut(&mut self.accelerations));
    }

    /// Computes the constant terms of the dynamics.
    fn update_dynamics<Bodies>(&mut self, dt: Real, bodies: &mut Bodies)
    where
        Bodies: ComponentSetMut<RigidBodyVelocity> + ComponentSet<RigidBodyMassProps>,
    {
        /*
         * Compute velocities.
         * NOTE: this is needed for kinematic bodies too.
         */
        let link = &mut self.links[0];
        let velocity_wrt_joint = link
            .articulation
            .jacobian_mul_coordinates(&self.velocities.as_slice()[link.assembly_id..]);
        let velocity_dot_wrt_joint = link
            .articulation
            .jacobian_dot_mul_coordinates(&self.velocities.as_slice()[link.assembly_id..]);

        link.velocity_dot_wrt_joint = velocity_dot_wrt_joint;
        link.velocity_wrt_joint = velocity_wrt_joint;
        bodies.set_internal(link.rigid_body.0, link.velocity_wrt_joint);

        for i in 1..self.links.len() {
            let (link, parent_link) = self.links.get_mut_with_parent(i);
            let rb_mprops: &RigidBodyMassProps = bodies.index(link.rigid_body.0);
            let (parent_rb_vels, parent_rb_mprops): (&RigidBodyVelocity, &RigidBodyMassProps) =
                bodies.index_bundle(parent_link.rigid_body.0);

            let velocity_wrt_joint = link
                .articulation
                .jacobian_mul_coordinates(&self.velocities.as_slice()[link.assembly_id..]);
            let velocity_dot_wrt_joint = link
                .articulation
                .jacobian_dot_mul_coordinates(&self.velocities.as_slice()[link.assembly_id..]);

            link.velocity_dot_wrt_joint =
                velocity_dot_wrt_joint.transformed(&parent_link.local_to_world);
            link.velocity_wrt_joint = velocity_wrt_joint.transformed(&parent_link.local_to_world);
            let mut new_rb_vels = *parent_rb_vels + link.velocity_wrt_joint;
            let shift = rb_mprops.world_com - parent_rb_mprops.world_com;
            new_rb_vels.linvel += parent_rb_vels.angvel.gcross(shift);

            bodies.set_internal(link.rigid_body.0, new_rb_vels);
        }

        /*
         * Update augmented mass matrix.
         */
        self.update_inertias(dt, bodies);
    }

    fn update_body_jacobians<Bodies>(&mut self, bodies: &Bodies)
    where
        Bodies: ComponentSet<RigidBodyMassProps>,
    {
        for i in 0..self.links.len() {
            let link = &self.links[i];
            let rb_mprops = bodies.index(link.rigid_body.0);

            if self.body_jacobians[i].ncols() != self.ndofs {
                // FIXME: use a resize instead.
                self.body_jacobians[i] = Jacobian::zeros(self.ndofs);
            }

            if i != 0 {
                let parent_id = link.parent_internal_id;
                let parent_link = &self.links[parent_id];
                let parent_rb_mprops = bodies.index(parent_link.rigid_body.0);

                let (link_j, parent_j) = self.body_jacobians.index_mut_const(i, parent_id);
                link_j.copy_from(&parent_j);

                {
                    let mut link_j_v = link_j.fixed_rows_mut::<DIM>(0);
                    let parent_j_w = parent_j.fixed_rows::<ANG_DIM>(DIM);

                    let shift_tr = (rb_mprops.world_com - parent_rb_mprops.world_com)
                        .gcross_matrix()
                        .transpose();
                    link_j_v.gemm(1.0, &shift_tr, &parent_j_w, 1.0);
                }
            } else {
                self.body_jacobians[i].fill(0.0);
            }

            let ndofs = link.articulation.ndofs();
            let mut tmp = SMatrix::<Real, SPATIAL_DIM, SPATIAL_DIM>::zeros();
            let mut link_joint_j = tmp.columns_mut(0, ndofs);
            let mut link_j_part = self.body_jacobians[i].columns_mut(link.assembly_id, ndofs);
            link.articulation
                .jacobian(&link.parent_to_world, &mut link_joint_j);

            link_j_part += link_joint_j;
        }
    }

    fn update_inertias<Bodies>(&mut self, dt: Real, bodies: &Bodies)
    where
        Bodies: ComponentSet<RigidBodyMassProps> + ComponentSet<RigidBodyVelocity>,
    {
        if self.augmented_mass.ncols() != self.ndofs {
            // TODO: do a resize instead of a full reallocation.
            self.augmented_mass = DMatrix::zeros(self.ndofs, self.ndofs);
        } else {
            self.augmented_mass.fill(0.0);
        }

        if self.coriolis_v.len() != self.links.len() {
            self.coriolis_v.resize(
                self.links.len(),
                OMatrix::<Real, Dim, Dynamic>::zeros(self.ndofs),
            );
            self.coriolis_w.resize(
                self.links.len(),
                OMatrix::<Real, AngDim, Dynamic>::zeros(self.ndofs),
            );
            self.i_coriolis_dt = Jacobian::zeros(self.ndofs);
        }

        for i in 0..self.links.len() {
            let link = &self.links[i];
            let (rb_vels, rb_mprops): (&RigidBodyVelocity, &RigidBodyMassProps) =
                bodies.index_bundle(link.rigid_body.0);
            let rb_mass = rb_mprops.effective_mass();
            let rb_inertia = rb_mprops.effective_angular_inertia().into_matrix();

            let body_jacobian = &self.body_jacobians[i];

            #[allow(unused_mut)] // mut is needed for 3D but not for 2D.
            let mut augmented_inertia = rb_inertia;

            #[cfg(feature = "dim3")]
            {
                // Derivative of gyroscopic forces.
                let gyroscopic_matrix = rb_vels.angvel.gcross_matrix() * rb_inertia
                    - (rb_inertia * rb_vels.angvel).gcross_matrix();

                augmented_inertia += gyroscopic_matrix * dt;
            }

            // FIXME: optimize that (knowing the structure of the augmented inertia matrix).
            // FIXME: this could be better optimized in 2D.
            let rb_mass_matrix = concat_rb_mass_matrix(rb_mass, augmented_inertia);
            self.augmented_mass
                .quadform(1.0, &rb_mass_matrix, body_jacobian, 1.0);

            /*
             *
             * Coriolis matrix.
             *
             */
            let rb_j = &self.body_jacobians[i];
            let rb_j_v = rb_j.fixed_rows::<DIM>(0);

            let ndofs = link.articulation.ndofs();

            if i != 0 {
                let parent_id = link.parent_internal_id;
                let parent_link = &self.links[parent_id];
                let (parent_rb_vels, parent_rb_mprops): (&RigidBodyVelocity, &RigidBodyMassProps) =
                    bodies.index_bundle(parent_link.rigid_body.0);
                let parent_j = &self.body_jacobians[parent_id];
                let parent_j_v = parent_j.fixed_rows::<DIM>(0);
                let parent_j_w = parent_j.fixed_rows::<ANG_DIM>(DIM);
                let parent_w = parent_rb_vels.angvel.gcross_matrix();

                let (coriolis_v, parent_coriolis_v) = self.coriolis_v.index_mut2(i, parent_id);
                let (coriolis_w, parent_coriolis_w) = self.coriolis_w.index_mut2(i, parent_id);

                // JDot + JDot/u * qdot
                coriolis_v.copy_from(&parent_coriolis_v);
                coriolis_w.copy_from(&parent_coriolis_w);

                let shift_tr = (rb_mprops.world_com - parent_rb_mprops.world_com)
                    .gcross_matrix()
                    .transpose();
                coriolis_v.gemm(1.0, &shift_tr, &parent_coriolis_w, 1.0);

                // JDot
                let dvel_tr = (rb_vels.linvel - parent_rb_vels.linvel)
                    .gcross_matrix()
                    .transpose();
                coriolis_v.gemm(1.0, &dvel_tr, &parent_j_w, 1.0);

                // JDot/u * qdot
                coriolis_v.gemm(
                    1.0,
                    &link.velocity_wrt_joint.linvel.gcross_matrix().transpose(),
                    &parent_j_w,
                    1.0,
                );
                coriolis_v.gemm(1.0, &parent_w, &rb_j_v, 1.0);
                coriolis_v.gemm(-1.0, &parent_w, &parent_j_v, 1.0);

                #[cfg(feature = "dim3")]
                {
                    let vel_wrt_joint_w = link.velocity_wrt_joint.angvel.gcross_matrix();
                    coriolis_w.gemm(-1.0, &vel_wrt_joint_w, &parent_j_w, 1.0);
                }

                {
                    let mut coriolis_v_part = coriolis_v.columns_mut(link.assembly_id, ndofs);

                    let mut tmp1 = SMatrix::<Real, SPATIAL_DIM, SPATIAL_DIM>::zeros();
                    let mut rb_joint_j = tmp1.columns_mut(0, ndofs);
                    link.articulation
                        .jacobian(&parent_link.local_to_world, &mut rb_joint_j);

                    let rb_joint_j_v = rb_joint_j.fixed_rows::<DIM>(0);

                    // JDot
                    coriolis_v_part.gemm(1.0, &parent_w, &rb_joint_j_v, 1.0);

                    #[cfg(feature = "dim3")]
                    {
                        let rb_joint_j_w = rb_joint_j.fixed_rows::<ANG_DIM>(DIM);
                        let mut coriolis_w_part = coriolis_w.columns_mut(link.assembly_id, ndofs);
                        coriolis_w_part.gemm(1.0, &parent_w, &rb_joint_j_w, 1.0);
                    }
                }
            } else {
                self.coriolis_v[i].fill(0.0);
                self.coriolis_w[i].fill(0.0);
            }

            let coriolis_v = &mut self.coriolis_v[i];
            let coriolis_w = &mut self.coriolis_w[i];

            {
                let mut tmp1 = SMatrix::<Real, SPATIAL_DIM, SPATIAL_DIM>::zeros();
                let mut tmp2 = SMatrix::<Real, SPATIAL_DIM, SPATIAL_DIM>::zeros();
                let mut rb_joint_j_dot = tmp1.columns_mut(0, ndofs);
                let mut rb_joint_j_dot_veldiff = tmp2.columns_mut(0, ndofs);

                link.articulation
                    .jacobian_dot(&link.parent_to_world, &mut rb_joint_j_dot);
                link.articulation.jacobian_dot_veldiff_mul_coordinates(
                    &link.parent_to_world,
                    &self.velocities.as_slice()[link.assembly_id..],
                    &mut rb_joint_j_dot_veldiff,
                );

                let rb_joint_j_v_dot = rb_joint_j_dot.fixed_rows::<DIM>(0);
                let rb_joint_j_w_dot = rb_joint_j_dot.fixed_rows::<ANG_DIM>(DIM);
                let rb_joint_j_v_dot_veldiff = rb_joint_j_dot_veldiff.fixed_rows::<DIM>(0);
                let rb_joint_j_w_dot_veldiff = rb_joint_j_dot_veldiff.fixed_rows::<ANG_DIM>(DIM);

                let mut coriolis_v_part = coriolis_v.columns_mut(link.assembly_id, ndofs);
                let mut coriolis_w_part = coriolis_w.columns_mut(link.assembly_id, ndofs);

                // JDot
                coriolis_v_part += rb_joint_j_v_dot;
                coriolis_w_part += rb_joint_j_w_dot;

                // JDot/u * qdot
                coriolis_v_part += rb_joint_j_v_dot_veldiff;
                coriolis_w_part += rb_joint_j_w_dot_veldiff;
            }

            /*
             * Meld with the mass matrix.
             */
            {
                let mut i_coriolis_dt_v = self.i_coriolis_dt.fixed_rows_mut::<DIM>(0);
                i_coriolis_dt_v.copy_from(coriolis_v);
                i_coriolis_dt_v *= rb_mass * dt;
            }

            {
                // FIXME: in 2D this is just an axpy.
                let mut i_coriolis_dt_w = self.i_coriolis_dt.fixed_rows_mut::<ANG_DIM>(DIM);
                i_coriolis_dt_w.gemm(dt, &rb_inertia, &coriolis_w, 0.0);
            }

            self.augmented_mass
                .gemm_tr(1.0, &rb_j, &self.i_coriolis_dt, 1.0);
        }

        /*
         * Damping.
         */
        for i in 0..self.ndofs {
            self.augmented_mass[(i, i)] += self.damping[i] * dt;
        }

        // FIXME: avoid allocation inside LU at each timestep.
        self.inv_augmented_mass = LU::new(self.augmented_mass.clone());
    }

    /*
    /// The generalized velocity at the articulation of the given link.
    #[inline]
    pub fn articulation_velocity(&self, link: &MultibodyLink) -> DVectorSlice<Real> {
        let ndofs = link.dof.ndofs();
        DVectorSlice::from_slice(
            &self.velocities.as_slice()[link.assembly_id..link.assembly_id + ndofs],
            ndofs,
        )
    }

    /// Convert a force applied to the center of mass of the link `rb_id` into generalized force.
    pub fn link_jacobian_mul_force(
        &self,
        link: &MultibodyLink,
        force: &Force<Real>,
        out: &mut [Real],
    ) {
        let mut out = DVectorSliceMut::from_slice(out, self.ndofs);
        self.body_jacobians[link.internal_id].tr_mul_to(force.as_vector(), &mut out);
    }

    /// Convert a force applied to this multibody's link `rb_id` center of mass into generalized accelerations.
    pub fn inv_mass_mul_link_force(
        &self,
        link: &MultibodyLink,
        force: &Force<Real>,
        out: &mut [Real],
    ) {
        let mut out = DVectorSliceMut::from_slice(out, self.ndofs);
        self.body_jacobians[link.internal_id].tr_mul_to(force.as_vector(), &mut out);
        assert!(self.inv_augmented_mass.solve_mut(&mut out));
    }

    /// Convert a generalized force applied to le link `rb_id`'s degrees of freedom into generalized accelerations.
    ///
    /// The joint attaching this link to its parent is assumed to be a unit joint.
    pub fn inv_mass_mul_unit_joint_force(
        &self,
        link: &MultibodyLink,
        dof_id: usize,
        force: Real,
        out: &mut [Real],
    ) {
        let mut out = DVectorSliceMut::from_slice(out, self.ndofs);
        out.fill(0.0);
        out[link.assembly_id + dof_id] = force;
        assert!(self.inv_augmented_mass.solve_mut(&mut out));
    }

    /// Convert a generalized force applied to the link `rb_id`'s degrees of freedom into generalized accelerations.
    pub fn inv_mass_mul_joint_force(
        &self,
        link: &MultibodyLink,
        force: DVectorSlice<Real>,
        out: &mut [Real],
    ) {
        let ndofs = link.dof.ndofs();

        let mut out = DVectorSliceMut::from_slice(out, self.ndofs);
        out.fill(0.0);
        out.rows_mut(link.assembly_id, ndofs).copy_from(&force);
        assert!(self.inv_augmented_mass.solve_mut(&mut out));
    }

    /// The augmented mass (inluding gyroscropic and coriolis terms) in world-space of this multibody.
    pub fn augmented_mass(&self) -> &DMatrix<Real> {
        &self.augmented_mass
    }

    /// Retrieve the mutable generalized velocities of this link.
    #[inline]
    pub fn joint_velocity_mut(&mut self, id: usize) -> DVectorSliceMut<Real> {
        let ndofs;
        let i;
        {
            let link = self.link(id).expect("Invalid multibody link handle.");
            ndofs = link.dof.ndofs();
            i = link.assembly_id;
        }

        DVectorSliceMut::from_slice(&mut self.velocities.as_mut_slice()[i..i + ndofs], ndofs)
    }

    #[inline]
    pub fn generalized_acceleration(&self) -> DVectorSlice<Real> {
        self.accelerations.rows(0, self.ndofs)
    }

    #[inline]
    pub fn generalized_velocity(&self) -> DVectorSlice<Real> {
        self.velocities.rows(0, self.ndofs)
    }

    #[inline]
    pub fn generalized_velocity_mut(&mut self) -> DVectorSliceMut<Real> {
        self.velocities.rows_mut(0, self.ndofs)
    }

    #[inline]
    pub(crate) fn impulses(&self) -> &[Real] {
        self.impulses.as_slice()
    }

    #[inline]
    pub fn integrate(&mut self, parameters: &IntegrationParameters) {
        for rb in self.links.iter_mut() {
            rb.articulation
                .integrate(parameters, &self.velocities.as_slice()[rb.assembly_id..])
        }
    }

    pub fn apply_displacement(&mut self, disp: &[Real]) {
        for rb in self.links.iter_mut() {
            rb.articulation.apply_displacement(&disp[rb.assembly_id..])
        }

        self.update_kinematics();
    }
     */

    pub fn forward_kinematics<Bodies>(&mut self, bodies: &mut Bodies)
    where
        Bodies: ComponentSetMut<RigidBodyMassProps> + ComponentSetMut<RigidBodyPosition>,
    {
        // Special case for the root, which has no parent.
        {
            let link = &mut self.links[0];
            link.articulation
                .update_jacobians(&link.body_shift, &self.velocities.as_slice());
            link.local_to_parent = link
                .articulation
                .body_to_parent(&link.parent_shift, &link.body_shift);
            link.local_to_world = link.local_to_parent;

            bodies.set_internal(
                link.rigid_body.0,
                RigidBodyPosition::from(link.local_to_world),
            );

            bodies.map_mut_internal(link.rigid_body.0, |mprops: &mut RigidBodyMassProps| {
                mprops.update_world_mass_properties(&link.local_to_world)
            });
        }

        // Handle the children. They all have a parent within this multibody.
        for i in 1..self.links.len() {
            let (link, parent_link) = self.links.get_mut_with_parent(i);

            link.articulation.update_jacobians(
                &link.body_shift,
                &self.velocities.as_slice()[link.assembly_id..],
            );
            link.local_to_parent = link
                .articulation
                .body_to_parent(&link.parent_shift, &link.body_shift);
            link.local_to_world = parent_link.local_to_world * link.local_to_parent;
            link.parent_to_world = parent_link.local_to_world;

            bodies.set_internal(
                link.rigid_body.0,
                RigidBodyPosition::from(link.local_to_world),
            );

            bodies.map_mut_internal(link.rigid_body.0, |mprops: &mut RigidBodyMassProps| {
                mprops.update_world_mass_properties(&link.local_to_world)
            });
        }

        /*
         * Compute body jacobians.
         */
        self.update_body_jacobians(bodies);
    }

    /*
    #[inline]
    pub fn ndofs(&self) -> usize {
        self.ndofs
    }

    pub fn fill_constraint_geometry(
        &self,
        link: &MultibodyLink,
        ndofs: usize, // FIXME: keep this parameter?
        point: &Point<Real>,
        force_dir: &ForceDirection<Real>,
        j_id: usize,
        wj_id: usize,
        jacobians: &mut [Real],
        inv_r: &mut Real,
        ext_vels: Option<&DVectorSlice<Real>>,
        out_vel: Option<&mut Real>,
    ) {
        let pos = point - link.com.coords;
        let force = force_dir.at_point(&pos);

        match self.status() {
            RigidBodyType::Dynamic => {
                self.link_jacobian_mul_force(link, &force, &mut jacobians[j_id..]);

                // FIXME: this could be optimized with a copy_nonoverlapping.
                for i in 0..ndofs {
                    jacobians[wj_id + i] = jacobians[j_id + i];
                }

                {
                    let mut out = DVectorSliceMut::from_slice(&mut jacobians[wj_id..], self.ndofs);
                    assert!(self.inv_augmented_mass.solve_mut(&mut out))
                }

                let j = DVectorSlice::from_slice(&jacobians[j_id..], ndofs);
                let invm_j = DVectorSlice::from_slice(&jacobians[wj_id..], ndofs);

                *inv_r += j.dot(&invm_j);

                if let Some(out_vel) = out_vel {
                    *out_vel += j.dot(&self.generalized_velocity());

                    if let Some(ext_vels) = ext_vels {
                        *out_vel += j.dot(ext_vels)
                    }
                }
            }
            RigidBodyType::Kinematic => {
                if let Some(out_vel) = out_vel {
                    *out_vel += force.as_vector().dot(&link.velocity.as_vector())
                }
            }
            RigidBodyType::Static => {}
        }
    }

     */

    /*
    #[inline]
    pub fn has_active_internal_constraints(&mut self) -> bool {
        self.links()
            .any(|link| link.joint().num_velocity_constraints() != 0)
    }

    #[inline]
    pub fn setup_internal_velocity_constraints(
        &mut self,
        ext_vels: &DVectorSlice<Real>,
        parameters: &IntegrationParameters,
    ) {
        let mut ground_j_id = 0;
        let mut workspace = self.solver_workspace.take().unwrap();

        /*
         * Cache impulses from the last timestep for warmstarting.
         */
        // FIXME: should this be another pass of the solver (happening after all the resolution completed).
        for c in &workspace.constraints.velocity.unilateral_ground {
            self.impulses[c.impulse_id] = c.impulse;
        }

        for c in &workspace.constraints.velocity.bilateral_ground {
            self.impulses[c.impulse_id] = c.impulse;
        }

        workspace.constraints.clear();

        /*
         * Setup the constraints.
         */
        let nconstraints = self
            .rbs
            .iter()
            .map(|l| l.joint().num_velocity_constraints())
            .sum();

        workspace.resize(nconstraints, self.ndofs);

        for link in self.links.iter() {
            link.joint().velocity_constraints(
                parameters,
                self,
                &link,
                0,
                0,
                ext_vels.as_slice(),
                &mut ground_j_id,
                workspace.jacobians.as_mut_slice(),
                &mut workspace.constraints,
            );
        }

        self.solver_workspace = Some(workspace);
    }

    #[inline]
    pub fn warmstart_internal_velocity_constraints(&mut self, dvels: &mut DVectorSliceMut<Real>) {
        let workspace = self.solver_workspace.as_mut().unwrap();
        for c in &mut workspace.constraints.velocity.unilateral_ground {
            let dim = Dynamic::new(c.ndofs);
            SORProx::warmstart_unilateral_ground(c, workspace.jacobians.as_slice(), dvels, dim)
        }

        for c in &mut workspace.constraints.velocity.bilateral_ground {
            let dim = Dynamic::new(c.ndofs);
            SORProx::warmstart_bilateral_ground(c, workspace.jacobians.as_slice(), dvels, dim)
        }
    }

    #[inline]
    pub fn step_solve_internal_velocity_constraints(&mut self, dvels: &mut DVectorSliceMut<Real>) {
        let workspace = self.solver_workspace.as_mut().unwrap();
        for c in &mut workspace.constraints.velocity.unilateral_ground {
            let dim = Dynamic::new(c.ndofs);
            SORProx::solve_unilateral_ground(c, workspace.jacobians.as_slice(), dvels, dim)
        }

        for c in &mut workspace.constraints.velocity.bilateral_ground {
            let dim = Dynamic::new(c.ndofs);
            SORProx::solve_bilateral_ground(c, &[], workspace.jacobians.as_slice(), dvels, dim)
        }
    }

    #[inline]
    pub fn step_solve_internal_position_constraints(&mut self, parameters: &IntegrationParameters) {
        // FIXME: this `.take()` trick is ugly.
        // We should not pass a reference to the multibody to the link position constraint method.
        let mut workspace = self.solver_workspace.take().unwrap();
        let jacobians = &mut workspace.jacobians;

        for i in 0..self.links.len() {
            for j in 0..self.links[i].joint().num_position_constraints() {
                let link = &self.links[i];
                // FIXME: should each link directly solve the constraint internally
                // instead of having to return a GenericNonlinearConstraint struct
                // every time.
                let c = link.joint().position_constraint(
                    j,
                    self,
                    link,
                    BodyPartHandle((), i),
                    0,
                    jacobians.as_mut_slice(),
                );

                if let Some(c) = c {
                    // FIXME: the following has been copy-pasted from the NonlinearSORProx.
                    // We should refactor the code better.
                    let rhs = NonlinearSORProx::clamp_rhs(c.rhs, c.is_angular, parameters);

                    if rhs < 0.0 {
                        let impulse = -rhs * c.r;
                        jacobians.rows_mut(c.wj_id1, c.dim1).mul_assign(impulse);

                        // FIXME: we should not use apply_displacement to avoid
                        // performing the .update_kinematic().
                        self.apply_displacement(jacobians.rows(c.wj_id1, c.dim1).as_slice());
                    }
                }
            }
        }
        self.solver_workspace = Some(workspace);
        self.update_kinematics();
    }
    */
}
