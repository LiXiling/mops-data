/**
 * Three.js scene setup and rendering for VR
 */
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.174.0/build/three.module.js';
import { debugLog } from './utils.js';

// Three.js objects
export let renderer, scene, camera;
let cube, cubeLeft, cubeRight;

// Debug visualization objects
let leftTrail = [];
let rightTrail = [];
const TRAIL_LENGTH = 20; // Number of points in each trail
const TRAIL_INTERVAL = 5; // Frames between trail updates

// We'll need to reference these from webxr.js, but avoid circular dependencies
export let leftController = null;
export let rightController = null;

/**
 * Initialize Three.js scene, camera, renderer and objects
 */
export function initRenderer(elements) {
    debugLog("Initializing Three.js renderer with passthrough support...");

    // Create renderer with alpha support for passthrough
    renderer = new THREE.WebGLRenderer({
        canvas: elements.canvas,
        antialias: true,
        alpha: true,  // Enable alpha for passthrough
        premultipliedAlpha: false  // Better for passthrough
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.xr.enabled = true;

    // Set clear color with alpha=0 for passthrough
    renderer.setClearColor(0x000000, 0);

    // Create scene
    scene = new THREE.Scene();
    // Start with transparent background for passthrough
    scene.background = null;

    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);

    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1).normalize();
    scene.add(directionalLight);

    // Create camera
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 3;

    // Create main spinning cube (visible in non-VR mode)
    const geometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const material = new THREE.MeshStandardMaterial({
        color: 0x5555ff,
        metalness: 0.3,
        roughness: 0.4,
    });
    cube = new THREE.Mesh(geometry, material);
    cube.position.set(0, 1.6, -2); // Position in front of the user
    scene.add(cube);

    // Create cubes for the controllers
    const controllerGeometry = new THREE.BoxGeometry(0.15, 0.15, 0.15);
    const leftMaterial = new THREE.MeshStandardMaterial({ color: 0xff0000 });
    const rightMaterial = new THREE.MeshStandardMaterial({ color: 0x00ff00 });

    cubeLeft = new THREE.Mesh(controllerGeometry, leftMaterial);
    cubeRight = new THREE.Mesh(controllerGeometry, rightMaterial);

    scene.add(cubeLeft);
    scene.add(cubeRight);

    // Initialize controller trails
    initControllerTrails();
    createCoordinateAxes();

    // Handle window resize
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    debugLog("Three.js renderer initialized successfully");
    return renderer;
}

/**
 * Initialize controller movement trails
 */
function initControllerTrails() {
    debugLog("Initializing controller trails...");

    // Create trail objects for left controller
    for (let i = 0; i < TRAIL_LENGTH; i++) {
        const trailGeometry = new THREE.BoxGeometry(0.02, 0.02, 0.02);
        const trailMaterial = new THREE.MeshBasicMaterial({
            color: 0xff6666,
            transparent: true,
            opacity: 1 - (i / TRAIL_LENGTH) // Fade out older trail points
        });

        const trailCube = new THREE.Mesh(trailGeometry, trailMaterial);
        trailCube.visible = false;
        scene.add(trailCube);
        leftTrail.push(trailCube);
    }

    // Create trail objects for right controller
    for (let i = 0; i < TRAIL_LENGTH; i++) {
        const trailGeometry = new THREE.BoxGeometry(0.02, 0.02, 0.02);
        const trailMaterial = new THREE.MeshBasicMaterial({
            color: 0x66ff66,
            transparent: true,
            opacity: 1 - (i / TRAIL_LENGTH) // Fade out older trail points
        });

        const trailCube = new THREE.Mesh(trailGeometry, trailMaterial);
        trailCube.visible = false;
        scene.add(trailCube);
        rightTrail.push(trailCube);
    }
}

/**
 * Set controller references
 * This function is used to avoid circular imports
 */
export function setControllerReferences(left, right) {
    leftController = left;
    rightController = right;
}

/**
 * Update controller visualization in the scene
 * @param {XRFrame} frame - The current XR frame
 * @param {XRReferenceSpace} referenceSpace - The reference space
 */
export function updateControllerVisualization(frame, referenceSpace) {
    // Get trigger values and adjust rotation speed
    let rotationSpeed = 0.01; // Default speed
    let leftTriggerValue = 0;
    let rightTriggerValue = 0;

    if (leftController && leftController.gamepad && leftController.gamepad.buttons[0]) {
        leftTriggerValue = leftController.gamepad.buttons[0].value || 0;
    }

    if (rightController && rightController.gamepad && rightController.gamepad.buttons[0]) {
        rightTriggerValue = rightController.gamepad.buttons[0].value || 0;
    }

    // Adjust rotation speed based on trigger pressure (average of both triggers)
    const triggerAverage = (leftTriggerValue + rightTriggerValue) / 2;
    rotationSpeed = 0.01 + (triggerAverage * 0.1); // Scale up to 10x faster with full trigger press

    // Rotate the main cube with speed based on trigger pressure
    if (cube) {
        cube.rotation.x += rotationSpeed;
        cube.rotation.y += rotationSpeed;
    }

    // Update controller debug trail
    updateControllerTrail(frame, referenceSpace);

    // Update left controller visualization
    if (leftController) {
        const pose = frame.getPose(leftController.gripSpace, referenceSpace);
        if (pose && cubeLeft) {
            cubeLeft.visible = true;
            cubeLeft.position.set(
                pose.transform.position.x,
                pose.transform.position.y,
                pose.transform.position.z
            );
            cubeLeft.quaternion.set(
                pose.transform.orientation.x,
                pose.transform.orientation.y,
                pose.transform.orientation.z,
                pose.transform.orientation.w
            );

            // Change left cube color based on trigger pressure
            cubeLeft.material.color.setRGB(1, 1 - leftTriggerValue, 1 - leftTriggerValue);
        }
    } else {
        if (cubeLeft) cubeLeft.visible = false;
    }

    // Update right controller visualization
    if (rightController) {
        const pose = frame.getPose(rightController.gripSpace, referenceSpace);
        if (pose && cubeRight) {
            cubeRight.visible = true;
            cubeRight.position.set(
                pose.transform.position.x,
                pose.transform.position.y,
                pose.transform.position.z
            );
            cubeRight.quaternion.set(
                pose.transform.orientation.x,
                pose.transform.orientation.y,
                pose.transform.orientation.z,
                pose.transform.orientation.w
            );

            // Change right cube color based on trigger pressure
            cubeRight.material.color.setRGB(1 - rightTriggerValue, 1, 1 - rightTriggerValue);
        }
    } else {
        if (cubeRight) cubeRight.visible = false;
    }
}

/**
 * Update the controller trail visualization
 * @param {XRFrame} frame - The current XR frame
 * @param {XRReferenceSpace} referenceSpace - The reference space
 */
function updateControllerTrail(frame, referenceSpace) {
    // Only update the trail every TRAIL_INTERVAL frames
    if (frame.frameCount % TRAIL_INTERVAL !== 0) {
        return;
    }

    // Update left controller trail
    if (leftController) {
        const pose = frame.getPose(leftController.gripSpace, referenceSpace);
        if (pose) {
            // Move all trail points back one position
            for (let i = leftTrail.length - 1; i > 0; i--) {
                leftTrail[i].position.copy(leftTrail[i - 1].position);
                leftTrail[i].visible = leftTrail[i - 1].visible;
            }

            // Add new position at the front of the trail
            leftTrail[0].position.set(
                pose.transform.position.x,
                pose.transform.position.y,
                pose.transform.position.z
            );
            leftTrail[0].visible = true;
        }
    }

    // Update right controller trail
    if (rightController) {
        const pose = frame.getPose(rightController.gripSpace, referenceSpace);
        if (pose) {
            // Move all trail points back one position
            for (let i = rightTrail.length - 1; i > 0; i--) {
                rightTrail[i].position.copy(rightTrail[i - 1].position);
                rightTrail[i].visible = rightTrail[i - 1].visible;
            }

            // Add new position at the front of the trail
            rightTrail[0].position.set(
                pose.transform.position.x,
                pose.transform.position.y,
                pose.transform.position.z
            );
            rightTrail[0].visible = true;
        }
    }
}

/**
 * Start the animation loop
 * @param {function} xrFrameCallback - Callback for XR frame updates
 */
export function startAnimationLoop(xrFrameCallback) {
    let frameCount = 0;

    renderer.setAnimationLoop((time, frame) => {
        // Increment frame counter
        frameCount++;

        // Add frameCount to the frame object for trail timing
        if (frame) {
            frame.frameCount = frameCount;
        }

        // Non-XR mode: just rotate the cube
        if (!renderer.xr.isPresenting) {
            if (cube) {
                cube.rotation.x += 0.01;
                cube.rotation.y += 0.01;
            }

            // Render scene (non-XR)
            renderer.render(scene, camera);
        }
        // XR mode: handle controller tracking and VR rendering
        else if (frame) {
            xrFrameCallback(time, frame);

            // Render scene (XR)
            renderer.render(scene, camera);
        }
    });
}

/**
 * Create coordinate axes visualization
 * This helps users understand the coordinate system
 */
export function createCoordinateAxes() {
    const axesLength = 0.5;
    const axesWidth = 0.005;

    // Create X axis (red)
    const xGeometry = new THREE.BoxGeometry(axesLength, axesWidth, axesWidth);
    const xMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
    const xAxis = new THREE.Mesh(xGeometry, xMaterial);
    xAxis.position.set(axesLength / 2, 0, 0);
    scene.add(xAxis);

    // Create Y axis (green)
    const yGeometry = new THREE.BoxGeometry(axesWidth, axesLength, axesWidth);
    const yMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    const yAxis = new THREE.Mesh(yGeometry, yMaterial);
    yAxis.position.set(0, axesLength / 2, 0);
    scene.add(yAxis);

    // Create Z axis (blue)
    const zGeometry = new THREE.BoxGeometry(axesWidth, axesWidth, axesLength);
    const zMaterial = new THREE.MeshBasicMaterial({ color: 0x0000ff });
    const zAxis = new THREE.Mesh(zGeometry, zMaterial);
    zAxis.position.set(0, 0, axesLength / 2);
    scene.add(zAxis);

    // Create text labels
    addAxisLabel("X", axesLength, 0, 0, 0xff0000);
    addAxisLabel("Y", 0, axesLength, 0, 0x00ff00);
    addAxisLabel("Z", 0, 0, axesLength, 0x0000ff);

    return { xAxis, yAxis, zAxis };
}

/**
 * Add text label for coordinate axis
 */
function addAxisLabel(text, x, y, z, color) {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;

    const context = canvas.getContext('2d');
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, canvas.width, canvas.height);

    context.font = 'bold 80px Arial';
    context.fillStyle = `#${color.toString(16).padStart(6, '0')}`;
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(text, canvas.width / 2, canvas.height / 2);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(material);

    sprite.position.set(x, y, z);
    sprite.scale.set(0.1, 0.1, 0.1);
    scene.add(sprite);
}
