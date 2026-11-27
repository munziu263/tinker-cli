# The viewer's one dependency

`three-bundle.js` is three.js r170 with the two add-ons the page uses, as a single ES
module, so that the page loads nothing from anywhere else. It is made from a three-line
entry file:

```js
// entry.js
export * as THREE from 'three';
export { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
export { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
```

```
npm install three@0.170.0 esbuild@0.24.0
npx esbuild entry.js --bundle --format=esm --minify --legal-comments=eof --outfile=three-bundle.js
cp node_modules/three/LICENSE THREE-LICENSE.txt
```

To move to a newer three.js, change the version, rebuild, and check the viewer in every
mode: lights, shadows and colour handling are the things that shift between releases.
