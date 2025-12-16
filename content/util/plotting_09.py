import matplotlib.pyplot as plt

def  plot_09(iterates, image_path):
    pic = plt.imread(image_path)
    fig = plt.figure(figsize = (30, 15))
    plt.imshow(pic)
    plt.axis('off')

    # Diese Werte sind heuristisch bestimmt, da
    # nicht völlig klar ist, welches Koordinatensystem
    # in der Tabelle verwendet wird, und wie es sich 
    # in die Pixelwerte der Karte übersetzen lässt
    skalax = 206.0 / 3200
    skalay = 228.0 / 3590
    #
    xkoords = skalax * iterates[:,0] + 23.0
    ykoords = 892.0 - skalay * iterates[:,1]
    #
    plt.plot(xkoords, ykoords, 'k--', linewidth = 2)
    plt.plot(xkoords[:-1], ykoords[:-1], 'kx', markersize = 10)
    plt.plot(xkoords[-1], ykoords[-1], 'ro', markersize = 10)

    plt.arrow(xkoords[-1], ykoords[-1], iterates[-1,3]*300, -iterates[-1,4]*300, width = 3, color = 'blue')
    
    plt.show()