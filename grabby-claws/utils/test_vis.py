import trimesh

m = trimesh.load("frame_0000 (1).obj")
left = trimesh.Scene()
left.add_geometry(m.split()[-1])
m1 = m.split()[-3]
left.add_geometry(m1)
m2 = m.split()[-2]
left.export("left.obj")
