import glob
from PIL import Image
import matplotlib.pyplot as plt


SNIPER = pd.read_csv(cwd + '/SNIPER_Output/SNIPER_OUTPUT.csv')
for object in SNIPER['TNS Name']:
    # Use glob to find all PNG files in the current directory
    png_files = glob.glob(cwd + '/SNIPER_Output/'+object+'*')

    print(png_files)

    # Filter out the first two PNG files (you can adjust this based on your needs)
    png_files = png_files[:2]

    # Load the images
    images = [Image.open(png_file) for png_file in png_files]

    # Calculate the total width for the side-by-side display
    total_width = sum(image.width for image in images)

    # Calculate the maximum height among the images
    max_height = max(image.height for image in images)

    # Create a new image with the calculated width and maximum height
    combined_image = Image.new('RGB', (total_width, max_height))

    # Paste the images side by side onto the new image
    x_offset = 0
    for image in images:
        combined_image.paste(image, (x_offset, 0))
        x_offset += image.width

    # Show the combined image
    plt.imshow(combined_image)
    plt.axis('off')  # Turn off axis
    plt.show()
