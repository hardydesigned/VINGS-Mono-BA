import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// ---- ENU(e,n,u) -> three (X=east, Y=up, Z=-north) -------------------------
const E = (e, n, u) => new THREE.Vector3(e, u, -n);

const params = new URLSearchParams(location.search);
const VIEW = params.get('view') || 'top';   // top | persp | obl
const PARAM = (k, d) => params.has(k) ? params.get(k) : d;
const ON = (k, d) => PARAM(k, d) !== '0';
const setStatus = (s) => { document.getElementById('status').textContent = s; };

const CLASS_COLOR = { car: 0x35c2ff, van: 0x4be37a, truck: 0xffb547, bus: 0xff5d73 };
// model-space target dimensions in metres (length(forward) x width x height)
const VEH = {
  car:   { model: 'car.glb',   L: 4.6, W: 1.9, H: 1.6 },
  van:   { model: 'car.glb',   L: 5.4, W: 2.0, H: 2.2 },
  truck: { model: 'truck.glb', L: 7.5, W: 2.6, H: 3.0 },
  bus:   { model: 'bus.glb',   L: 11.0, W: 2.8, H: 3.3 },
};

const app = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x0b0e13, 1);
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.add(new THREE.AmbientLight(0xffffff, 1.4));
const sun = new THREE.DirectionalLight(0xffffff, 1.8);
sun.position.set(0.4, 1, 0.25);
scene.add(sun);

let camera, controls;

function makeCamera(center, radius) {
  const aspect = innerWidth / innerHeight;
  const zoom = +PARAM('zoom', 1);
  radius = radius / zoom;
  // optional pan in metres (east,north)
  const pe = +PARAM('pe', 0), pn = +PARAM('pn', 0);
  center = center.clone(); center.x += pe; center.z += -pn;
  if (VIEW === 'top') {
    const h = radius * 1.12;
    camera = new THREE.OrthographicCamera(-h * aspect, h * aspect, h, -h, 0.1, 100000);
    camera.position.set(center.x, center.y + radius * 3, center.z + 0.001);
    camera.up.set(0, 0, -1);              // north up on screen
  } else {
    camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 200000);
    const d = radius * (VIEW === 'obl' ? 1.7 : 2.0);
    if (VIEW === 'obl') camera.position.set(center.x + d * 0.1, center.y + d * 0.9, center.z + d * 0.9);
    else camera.position.set(center.x, center.y + d, center.z + d * 0.02);
    camera.up.set(0, 1, 0);
  }
  camera.lookAt(center);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(center);
  controls.update();
}

// ---------------------------------------------------------------- main load
async function main() {
  const scn = await (await fetch('assets/scene.json')).json();
  const bb = scn.cloud_bbox;
  const center = E((bb.e[0] + bb.e[1]) / 2, (bb.n[0] + bb.n[1]) / 2, scn.ground_z);
  const radius = 0.5 * Math.hypot(bb.e[1] - bb.e[0], bb.n[1] - bb.n[0]);
  makeCamera(center, radius);

  // ---- satellite ground plane ----
  const sat = scn.satellite;
  const tex = await new THREE.TextureLoader().loadAsync('assets/' + sat.png);
  tex.colorSpace = THREE.SRGBColorSpace;
  const w = sat.e_e - sat.e_w;            // east span (m)
  const d = sat.n_n - sat.n_s;            // north span (m)
  const planeGeo = new THREE.PlaneGeometry(w, d);
  const planeMat = new THREE.MeshBasicMaterial({ map: tex });
  const plane = new THREE.Mesh(planeGeo, planeMat);
  plane.rotation.x = -Math.PI / 2;        // XY -> XZ (lies flat, faces up)
  // after rotateX(-90): geometry +Y(north/top of image) -> +Z(three). We want
  // image-north at -Z(=ENU north). rotate 180 about Y so top-of-image -> -Z.
  plane.rotation.z = Math.PI;
  plane.position.set((sat.e_w + sat.e_e) / 2, scn.ground_z - 0.05, -(sat.n_n + sat.n_s) / 2);
  plane.renderOrder = -1;
  plane.visible = ON('sat', '1');
  scene.add(plane);
  window.__plane = plane;

  // ---- gaussian point cloud ----
  const [posBuf, colBuf] = await Promise.all([
    fetch('assets/cloud_pos.f32').then(r => r.arrayBuffer()),
    fetch('assets/cloud_col.u8').then(r => r.arrayBuffer()),
  ]);
  const posEnu = new Float32Array(posBuf);
  const colU8 = new Uint8Array(colBuf);
  const N = posEnu.length / 3;
  const pos = new Float32Array(N * 3), col = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    pos[i*3] = posEnu[i*3];           // east -> x
    pos[i*3+1] = posEnu[i*3+2];       // up   -> y
    pos[i*3+2] = -posEnu[i*3+1];      // north-> -z
    col[i*3] = colU8[i*3] / 255;
    col[i*3+1] = colU8[i*3+1] / 255;
    col[i*3+2] = colU8[i*3+2] / 255;
  }
  const cgeo = new THREE.BufferGeometry();
  cgeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  cgeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const cmat = new THREE.PointsMaterial({ size: +PARAM('psize', 1.7), vertexColors: true, sizeAttenuation: true, transparent: true, opacity: +PARAM('copac', 1.0) });
  const cloud = new THREE.Points(cgeo, cmat);
  cloud.visible = ON('cloud', '1');
  scene.add(cloud);
  window.__cloud = cloud;

  // ---- flight path ----
  const pts = scn.cam_path.map(p => E(p[0], p[1], p[2]));
  const pathGeo = new THREE.BufferGeometry().setFromPoints(pts);
  const path = new THREE.Line(pathGeo, new THREE.LineBasicMaterial({ color: 0xffe14d }));
  path.visible = ON('path', '1');
  scene.add(path);
  window.__path = path;

  // ---- objects ----
  const objects = await (await fetch('assets/objects.json')).json();
  const loader = new GLTFLoader();
  const modelCache = {};
  async function getModel(file) {
    if (!modelCache[file]) modelCache[file] = loader.loadAsync('assets/models/' + file);
    return (await modelCache[file]).scene.clone(true);
  }
  const objGroup = new THREE.Group();
  objGroup.visible = ON('obj', '1');
  scene.add(objGroup);
  window.__objGroup = objGroup;

  for (const o of objects) {
    const spec = VEH[o.cls] || VEH.car;
    const color = CLASS_COLOR[o.cls] || 0xffffff;
    let node;
    try {
      const m = await getModel(spec.model);
      // normalise GLB to its bbox, align longest horizontal axis -> forward(+Z model)
      const box = new THREE.Box3().setFromObject(m);
      const size = new THREE.Vector3(); box.getSize(size);
      const cmid = new THREE.Vector3(); box.getCenter(cmid);
      m.position.sub(cmid);                       // centre at origin
      const pivot = new THREE.Group();
      pivot.add(m);
      // scale so x->W, y->H, z->L if that's the natural axis; pick longest horiz as length
      const horizLongAxis = size.x > size.z ? 'x' : 'z';
      const lenSrc = Math.max(size.x, size.z);
      const widSrc = Math.min(size.x, size.z);
      const sx = (horizLongAxis === 'x' ? spec.L : spec.W) / size.x;
      const sz = (horizLongAxis === 'x' ? spec.W : spec.L) / size.z;
      const sy = spec.H / size.y;
      m.scale.set(sx, sy, sz);
      if (horizLongAxis === 'x') m.rotation.y = Math.PI / 2; // make length point +Z
      node = pivot;
    } catch (e) {
      node = new THREE.Mesh(new THREE.BoxGeometry(spec.W, spec.H, spec.L),
                            new THREE.MeshStandardMaterial({ color }));
    }
    // ground objects: sit on the ground plane, lift by half height
    node.position.copy(E(o.x, o.y, scn.ground_z + spec.H / 2));
    node.rotation.y = o.yaw;                       // heading (N->E)
    objGroup.add(node);
    // ground ring sized to the vehicle (close-up marker)
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(spec.L * 0.55, spec.L * 0.72, 28),
      new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, transparent: true, opacity: 0.95 }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.copy(E(o.x, o.y, scn.ground_z + 0.15));
    objGroup.add(ring);
    // thin vertical beam so the object reads as a pin at full-scene zoom
    const beamH = 14;
    const beam = new THREE.Mesh(
      new THREE.CylinderGeometry(0.45, 0.45, beamH, 6),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.55 }));
    beam.position.copy(E(o.x, o.y, scn.ground_z + beamH / 2));
    objGroup.add(beam);
  }

  // legend
  const leg = document.getElementById('legend');
  leg.innerHTML = Object.entries(CLASS_COLOR).map(([k, v]) =>
    `<div><span style="background:#${v.toString(16).padStart(6,'0')}"></span>${k}</div>`).join('');

  // toggles
  const bind = (id, fn) => document.getElementById(id).addEventListener('change', e => { fn(e.target.checked); });
  bind('t-sat', v => plane.visible = v);
  bind('t-cloud', v => cloud.visible = v);
  bind('t-obj', v => objGroup.visible = v);
  bind('t-path', v => path.visible = v);
  document.getElementById('r-psize').addEventListener('input', e => cmat.size = +e.target.value);
  document.getElementById('r-copac').addEventListener('input', e => cmat.opacity = +e.target.value);

  setStatus(`${N.toLocaleString()} pts · ${objects.length} objects · view=${VIEW}`);
  window.__ready = true;
}

addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight);
  if (camera.isOrthographicCamera) {
    const a = innerWidth / innerHeight, h = (camera.top);
    camera.left = -h * a; camera.right = h * a; camera.updateProjectionMatrix();
  } else { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); }
});

function loop() {
  requestAnimationFrame(loop);
  if (!camera) return;
  controls && controls.update();
  renderer.render(scene, camera);
}
loop();
main().catch(e => { setStatus('ERROR: ' + e.message); console.error('MAIN FAILED', e, e.stack); window.__ready = true; });
