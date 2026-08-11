import numpy as np
import ipctk

root = "asserts/dlayers-demo/intesection_test"

vj = np.load(f"{root}/jacket/vertices.npy", mmap_mode="r")
vt = np.load(f"{root}/tee_sleeveless/vertices.npy", mmap_mode="r")
fj = np.load(f"{root}/jacket/faces.npy")
ft = np.load(f"{root}/tee_sleeveless/faces.npy")

V = np.vstack([vj[63], vt[63]])
F = np.vstack([fj, ft + vj.shape[1]])

E = np.concatenate([F[:,[0,1]], F[:,[1,2]], F[:,[2,0]]])
E = np.unique(np.sort(E, axis=1), axis=0)

def hit(V):
    V = np.asfortranarray(V, dtype=np.float64)
    mesh = ipctk.CollisionMesh(
        V,
        np.asfortranarray(E, dtype=np.int32),
        np.asfortranarray(F, dtype=np.int32),
    )
    return ipctk.has_intersections(mesh, V)

print("raw float64 :", hit(V))
print("float32 cast:", hit(V.astype(np.float32).astype(np.float64)))