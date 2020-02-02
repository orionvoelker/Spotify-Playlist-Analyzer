import spotipy
import json
import sys
import spotipy.util as util
from collections import Counter
import numpy as np
import cv2
from sklearn.cluster import KMeans
import imutils
import pprint
from matplotlib import pyplot as plt

from spotipy.oauth2 import SpotifyClientCredentials

        
artists_in_playlist = Counter()
artists_urls = []

def extractSkin(image):
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
    #lower_threshold = np.array([0,0,0], dtype=np.uint8)
    #upper_threshold = np.array([255,255,255], dtype=np.uint8)
    
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    skin = cv2.bitwise_and(img, img, mask=skinMask)

    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)

def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)

def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=True):

    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        # make the dictionay of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation

def extractDominantColor(image, hasThresholding=True, number_of_colors=10):#, hasThresholding=False):

    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding == True:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img)

    # Get Colour Information
    colorInformation = getColorInformation(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation

def plotColorBar(colorInformation):
    # Create a 500x100 black image
    color_bar = np.zeros((100, 500, 3), dtype="uint8")

    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int, (x['color'])))

        cv2.rectangle(color_bar, (int(top_x), 0),
                      (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar

def get_color_averages(color_info):
    red = 0
    green = 0
    blue = 0
    i = 0
    for x in color_info:
        red += x['color'][0]
        green += x['color'][1]
        blue += x['color'][2]
        i += 1
    red = red/i
    green = green/i
    blue = blue/i
    print ("R: %d G: %d B: %d" % (red, green, blue))
    return [float(red), float(green), float(blue)]
    
def prety_print_data(color_info):
    for x in color_info:
        print(pprint.pformat(x))
        print()

'''
Skin Image Primary : https://raw.githubusercontent.com/octalpixel/Skin-Extraction-from-Image-and-Finding-Dominant-Color/master/82764696-open-palm-hand-gesture-of-male-hand_image_from_123rf.com.jpg
Skin Image One     : https://raw.githubusercontent.com/octalpixel/Skin-Extraction-from-Image-and-Finding-Dominant-Color/master/skin.jpg
Skin Image Two     : https://raw.githubusercontent.com/octalpixel/Skin-Extraction-from-Image-and-Finding-Dominant-Color/master/skin_2.jpg
Skin Image Three   : https://raw.githubusercontent.com/octalpixel/Skin-Extraction-from-Image-and-Finding-Dominant-Color/master/Human-Hands-Front-Back-Image-From-Wikipedia.jpg
'''
        
def get_skin_colors():

    compiled_skins = []
    i = 0
    for url in range(len(artists_in_playlist)):
        image = imutils.url_to_image(artists_urls[url][1])
        #image = imutils.url_to_image('https://images-na.ssl-images-amazon.com/images/I/61n2ctN6jBL._AC_SX522_.jpg')
        
        # Resize image to a width of 250
        image = imutils.resize(image, width=250)
        
        # Show image
        #plt.subplot(3, 1, 1)
        #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #plt.title("Original Image")
        # plt.show()
        
        # Apply Skin Mask
        skin = extractSkin(image)
        
        #plt.subplot(3, 1, 2)
        #plt.imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
        #plt.title("Thresholded  Image")
        # plt.show()
        
        # Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of color
        dominantColors = extractDominantColor(skin, hasThresholding= True)
        
        # Show in the dominant color information
        print("Color Information")
        #$prety_print_data(dominantColors)
        rgb_avgs = get_color_averages(dominantColors)
        k = 3
        if ((rgb_avgs[0] >= 224.3 + (9.6*(-k))) and ((rgb_avgs[1] >= 193.1 + (17.0*(-k)))) and ((rgb_avgs[2] >= 177.6 + (21.0*(-k))))):
            compiled_skins.append([artists_urls[url][0], "white"])
            i+=1
        elif ((rgb_avgs[0] < 224.3 + (9.6*(-k))) and ((rgb_avgs[1] < 193.1 + (17.0*(-k)))) and ((rgb_avgs[2] < 177.6 + (21.0*(-k))))):
            compiled_skins.append([artists_urls[url][0], "black"])
            i+= 1
        # Show in the dominant color as bar
        #print("Color Bar")
        #colour_bar = plotColorBar(dominantColors)
        #plt.subplot(3, 1, 3)
        #plt.axis("off")
        #plt.imshow(colour_bar)
        #plt.title("Color Bar")
        
        #plt.tight_layout()
        #plt.show()
    return compiled_skins

def show_tracks(results):
    for i,item in enumerate(results['items']):
        track = item['track']
        test = (track['artists'][0]['name'])
        artists_in_playlist[test] += 1
        
def get_artist_pictures():
    for name in artists_in_playlist.keys():
        artist_pic = sp.search(q = 'artist:' + name, type = 'artist')
        items = artist_pic['artists']['items']
        if len(items) > 0:
            artist = items[0]
        artists_urls.append([artist['name'], artist['images'][0]['url']])

            
if len(sys.argv) > 1:
    username = sys.argv[1]

else:
    print "username or playlist error"
    
collab_or_private = input ("1 for collaborative playlist, 2 for private playlist: ")
choice = int(collab_or_private)

if (choice == 1):
    scope = 'playlist-read-private'
elif (choice == 2):
    scope = 'playlist-read-collaborative'
    
token = util.prompt_for_user_token(username, scope, client_id = '903b0af697414d8a972f39f4ee07823d',
                                   client_secret = '36ebe9260c014f5bae37d904e390016f',
                                   redirect_uri = 'http://localhost')

if token:
    sp = spotipy.Spotify(auth=token)
    playlist_id = raw_input ("Enter playlist uri (share -> Copy Spotify URI): ")

    playlist = sp.user_playlist(username, playlist_id, fields = "tracks,next")
    tracks = playlist['tracks']
    show_tracks(tracks)
    while tracks['next']:
        tracks = sp.next(tracks)
        show_tracks(tracks)
    for value in artists_in_playlist.keys():
        print "%s : %d" % (value, artists_in_playlist[value])
    get_artist_pictures()
    compiled_skins = get_skin_colors()
    i = 0
    total_white = 0
    total_black = 0
    for value in artists_in_playlist.keys():
        if (compiled_skins[i][1] == "black"):
            total_black += 1
        elif (compiled_skins[i][1] == "white"):
            total_white += 1
        print (compiled_skins[i], artists_in_playlist[value])
        i+=1
        
    print total_white
    print total_black
else:
    print "Can get token for", username

