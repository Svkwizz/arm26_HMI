import opensim as osim
def addActuatorsController(model):
    Frame = osim.PhysicalOffsetFrame('Frame',model.getBodySet().get(0), osim.Transform(osim.Vec3(0,-0.29,0)))
    elbow_wrap = osim.WrapCylinder()
    elbow_wrap.setName('elbow_wrap')                                           # Set name of the wrap object
    elbow_wrap.set_radius(0.045)                                               # Defining radius of the wrap cylinder
    elbow_wrap.set_length(0.02*2)                                              # Length of the cylinder
    elbow_wrap.set_quadrant('-x')                                              
    Frame.addWrapObject(elbow_wrap)                                            
    model.getBodySet().get(0).addComponent(Frame)
    
    # Adding Path actuators pact_a and pact_b
    pact_a = osim.PathActuator()                                 # Calling path actuator class
    pact_a.setName('pact_a')                                  # Name of the actuator
    pact_a.setOptimalForce(1)                               # Torque generated by torque actuator = (optimal force) x (control signal). In this case, optimal force = 1 is defined
    pact_a.addNewPathPoint('p2',model.getBodySet().get('armStrap'),osim.Vec3(0.04,0,0))           #Putting one anchor point of actuator over arm strap
    pact_a.addNewPathPoint('p3',model.getBodySet().get('forearmStrap'),osim.Vec3(0.02,0,-0.0))    # Another anchor point over forearm strap
    model.addForce(pact_a)
    pact_b = osim.PathActuator()                                 # Calling path actuator class
    pact_b.setName('pact_b')                                  # Name of the actuator
    pact_b.setOptimalForce(1)                               # Torque generated by torque actuator = (optimal force) x (control signal). In this case, optimal force = 1 is defined
    pact_b.addNewPathPoint('p5',model.getBodySet().get('armStrap'),osim.Vec3(-0.04,0,0))           # Putting one anchor point of actuator over arm strap
    pact_b.updGeometryPath().addPathWrap(elbow_wrap)
    pact_b.addNewPathPoint('p6',model.getBodySet().get('forearmStrap'),osim.Vec3(-0.02,0,-0.0))    # Another anchor point over forearm strap
    model.addComponent(pact_b)
    
    # Adding a 'brain' controller to excite muscles as per calculated muscle activations
    brain = osim.PrescribedController()
    for ctr in range(model.getMuscles().getSize()):
        brain.addActuator(model.getMuscles().get(ctr))
        
    brain.addActuator(pact_a)                                               # Adding the handle of the actuators to the controller 'brain'
    brain.addActuator(pact_b)
    model.addController(brain) 
    
    return [model,brain,pact_a,pact_b];