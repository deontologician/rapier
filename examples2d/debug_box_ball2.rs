use rapier2d::prelude::*;
use rapier_testbed2d::Testbed;

pub fn init_world(testbed: &mut Testbed) {
    /*
     * World
     */
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let joints = JointSet::new();

    /*
     * Ground
     */
    let rad = 1.0;
    let rigid_body = RigidBodyBuilder::new_static()
        .translation(vector![0.0, -rad])
        .rotation(std::f32::consts::PI / 4.0)
        .build();
    let handle = bodies.insert(rigid_body);
    let collider = ColliderBuilder::cuboid(rad, rad).build();
    colliders.insert_with_parent(collider, handle, &mut bodies);

    // Build the dynamic box rigid body.
    let rigid_body = RigidBodyBuilder::new_dynamic()
        .translation(vector![0.0, 3.0 * rad])
        .can_sleep(false)
        .build();
    let handle = bodies.insert(rigid_body);
    let collider = ColliderBuilder::ball(rad).build();
    colliders.insert_with_parent(collider, handle, &mut bodies);

    /*
     * Set up the testbed.
     */
    testbed.set_world(bodies, colliders, joints);
    testbed.look_at(point![0.0, 0.0], 50.0);
}
