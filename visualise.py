#!/usr/bin/env python

from vispy import app, visuals, gloo

class Visualiser(app.Canvas):
    """A vispy Canvas dedicated to showing off 2D arrays.
    Give it a callback that returns an array, and a wanted 
    framerate (# of frames per second), and it will display 
    what you need in a separate window.

    Note that it HAS to run in the main thread, so if you wanna do 
    anything else along the way, do it in a separate thread.
    """

    def __init__(self, image_callback, frame_rate, size=(800, 800)):
        app.Canvas.__init__(self, keys='interactive', size=size)
        self.image_callback = image_callback
        self.image = visuals.ImageVisual(image_callback(), method='subdivide')
        self.frame_time = 1./frame_rate
        
        # scale and center image in canvas
        s = 700. / max(self.image.size)
        t = 0.5 * (700. - (self.image.size[0] * s)) + 50
        self.image.transform = visuals.transforms.STTransform(scale=(s, s), translate=(t, 50))
        
        self.show()

    def on_draw(self, ev):
        self.image.set_data(self.image_callback())
        gloo.clear(color='black', depth=True)
        self.image.draw()

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.image.transforms.configure(canvas=self, viewport=vp)

    def run(self):
        t = app.Timer(0.1, connect=self.update, start=True)
        app.run()



if __name__ == '__main__':
    # Demonstration time!
    print("Demonstrating usage of visuals")
    print("Click the 'close' button on the opened window to stop this demo.")
    
    from threading import Thread
    import numpy as np

    class DummyGameManager(object):
        def __init__(self, size):
            self.size = size
            self.data = np.random.normal(size=size)

        def get_image(self):
            self.data += np.random.normal(scale=0.1, size=self.size)
            return self.data

    GM = DummyGameManager((15,15,3))
    win = Visualiser(GM.get_image, 2)
    win.run()