import processedImage

image_num = [57, 100, 200, 600,8000,2547,6004]


plotter = processedImage.ProcessedImagePlotter()

p_img = processedImage.ProcessedImage(57)
p_img2 = processedImage.ProcessedImage(257)

transform = processedImage.SteeredEdgeTransform(3,40,True)
transform2 = processedImage.CannyTransform(100,200,True)

plotter.plot_multy_grid([p_img,p_img2],[transform])