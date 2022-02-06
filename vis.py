import vispy
import yaml
import os
from vispy.scene import visuals, SceneCanvas
import numpy as np
from matplotlib import pyplot as plt
from transformer.laser import SemLaserScan
from test import predict

KITTI_CFG = yaml.safe_load(open('./config/kitti_simp.yaml', 'r'))
learning_map = np.unique(list(KITTI_CFG['learning_map'].values()))
nclasses = learning_map.size
color_dict = KITTI_CFG["color_map"]

scan = SemLaserScan(nclasses, color_dict, DATA=KITTI_CFG, H=33, W=513, fov_up=1.5, fov_down=-12.5, project=True)


class LaserScanVis:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, scan, scan_names, label_names, offset=0, predicitions=[], exec=False):
    self.scan = scan
    self.scan_names = scan_names
    self.label_names = label_names
    self.offset = offset
    self.total = len(self.scan_names)
    self.exec = exec
    self.predicitions  = predicitions
    self.reset()
    self.update_scan()

  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities

    # new canvas prepared for visualizing data
    self.canvas = SceneCanvas(keys='interactive', show=True)
    # interface (n next, b back, q quit, very simple)
    self.canvas.events.key_press.connect(self.key_press)
    self.canvas.events.draw.connect(self.draw)
    # grid
    self.grid = self.canvas.central_widget.add_grid()

    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.canvas.scene)
    self.grid.add_widget(self.scan_view, 0, 0)
    self.scan_vis = visuals.Markers()
    self.scan_view.camera = 'turntable'
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)
    # add semantics

    print("Using semantics in visualizer")
    self.sem_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.canvas.scene)
    self.grid.add_widget(self.sem_view, 0, 1)
    self.sem_vis = visuals.Markers()
    self.sem_view.camera = 'turntable'
    self.sem_view.add(self.sem_vis)
    visuals.XYZAxis(parent=self.sem_view.scene)



    # img canvas size
    self.multiplier = 1
    self.canvas_W = 513
    self.canvas_H = 33
    self.multiplier += 1

    # new canvas for img
    self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                  size=(self.canvas_W, self.canvas_H * self.multiplier))
    # grid
    self.img_grid = self.img_canvas.central_widget.add_grid()
    # interface (n next, b back, q quit, very simple)
    self.img_canvas.events.key_press.connect(self.key_press)
    self.img_canvas.events.draw.connect(self.draw)

    # add a view for the depth
    self.img_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.img_canvas.scene)
    self.img_grid.add_widget(self.img_view, 0, 0)
    self.img_vis = visuals.Image(cmap='viridis')
    self.img_view.add(self.img_vis)

    # add semantics
    self.sem_img_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.img_canvas.scene)
    self.img_grid.add_widget(self.sem_img_view, 1, 0)
    self.sem_img_vis = visuals.Image(cmap='viridis')
    self.sem_img_view.add(self.sem_img_vis)

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def update_scan(self):
    # first open data
    # import ipdb; ipdb.set_trace()
    self.scan.open_scan(self.scan_names[self.offset])
    if self.exec:
      self.scan.proj_sem_label = self.predicitions[self.offset]
      self.scan.inverse_map()
    else:
      self.scan.open_label(self.label_names[self.offset])
    self.scan.colorize(self.exec)

    # then change names
    title = "scan " + str(self.offset)
    self.canvas.title = title
    self.img_canvas.title = title

    # then do all the point cloud stuff

    # plot scan
    power = 16
    # print()
    range_data = np.copy(self.scan.unproj_range)
    # print(range_data.max(), range_data.min())
    range_data = range_data**(1 / power)
    # print(range_data.max(), range_data.min())
    viridis_range = ((range_data - range_data.min()) /
                     (range_data.max() - range_data.min()) *
                     255).astype(np.uint8)
    viridis_map = self.get_mpl_colormap("viridis")
    viridis_colors = viridis_map[viridis_range]

    if not self.exec:
      self.scan_vis.set_data(self.scan.points,
                            face_color=viridis_colors[..., ::-1],
                            edge_color=viridis_colors[..., ::-1],
                            size=1)

      # plot semantics
      self.sem_vis.set_data(self.scan.points,
                            face_color=self.scan.sem_label_color[..., ::-1],
                            edge_color=self.scan.sem_label_color[..., ::-1],
                            size=1)

    # now do all the range image stuff
    # plot range image
    data = np.copy(self.scan.proj_range)
    # print(data[data > 0].max(), data[data > 0].min())
    data[data > 0] = data[data > 0]**(1 / power)
    data[data < 0] = data[data > 0].min()
    # print(data.max(), data.min())
    data = (data - data[data > 0].min()) / \
        (data.max() - data[data > 0].min())
    # print(data.max(), data.min())
    self.img_vis.set_data(data)
    self.img_vis.update()

    self.sem_img_vis.set_data(self.scan.proj_sem_color[..., ::-1])
    self.sem_img_vis.update()

  # interface
  def key_press(self, event):
    self.canvas.events.key_press.block()
    self.img_canvas.events.key_press.block()
    if event.key == 'N':
      self.offset += 1
      if self.offset >= self.total:
        self.offset = 0
      self.update_scan()
    elif event.key == 'B':
      self.offset -= 1
      if self.offset < 0:
        self.offset = self.total - 1
      self.update_scan()
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()
    if self.img_canvas.events.key_press.blocked():
      self.img_canvas.events.key_press.unblock()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    self.img_canvas.close()
    vispy.app.quit()

  def run(self):
    vispy.app.run()

scan_paths = "/media/kitti/dataset/sequences/14/velodyne"
label_paths = "/media/kitti/dataset/sequences/00/labels"

# populate the pointclouds
scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(scan_paths)) for f in fn]
scan_names.sort()

label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(label_paths)) for f in fn]
label_names.sort()

predicitions = predict()

vis = LaserScanVis(scan=scan,
                   scan_names=scan_names,
                   label_names=label_names,
                   exec=True,
                   predicitions = predicitions)

# print instructions
print("To navigate:")
print("\tb: back (previous scan)")
print("\tn: next (next scan)")
print("\tq: quit (exit program)")

# run the visualizer
vis.run()