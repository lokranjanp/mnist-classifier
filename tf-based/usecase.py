import matplotlib.pyplot as plt

from main import *

model = tf.keras.models.load_model('handwritten.keras')

image_num = 1
while os.path.isfile(f"digits/digit{image_num}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_num}.png")[:,:,0]
        img = np.invert(np.array([img]))

        prediction = model.predict(img)
        print(f"This digit is probably {np.argmax(prediction)}")
        plt.imshow(img[0])
        plt.show()

    except:
        print("Error!")
    finally:
        image_num += 1