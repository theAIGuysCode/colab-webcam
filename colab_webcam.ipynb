{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "colab_webcam.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jzRKNe1iSlNW"
      },
      "source": [
        "# Google Colab: Access Webcam for Images and Video\r\n",
        "This notebook will go through how to access and run code on images and video taken using your webcam.  \r\n",
        "\r\n",
        "For this purpose of this tutorial we will be using OpenCV's Haar Cascade to do face detection on our Webcam image and video."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Fj9YcAnsT4B_"
      },
      "source": [
        "# import dependencies\r\n",
        "from IPython.display import display, Javascript, Image\r\n",
        "from google.colab.output import eval_js\r\n",
        "from base64 import b64decode, b64encode\r\n",
        "import cv2\r\n",
        "import numpy as np\r\n",
        "import PIL\r\n",
        "import io\r\n",
        "import html\r\n",
        "import time"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "L6pCmkJrUC9g"
      },
      "source": [
        "## Helper Functions\r\n",
        "Below are a few helper function to make converting between different image data types and formats. "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "09b_0FAnUa9y"
      },
      "source": [
        "# function to convert the JavaScript object into an OpenCV image\r\n",
        "def js_to_image(js_reply):\r\n",
        "  \"\"\"\r\n",
        "  Params:\r\n",
        "          js_reply: JavaScript object containing image from webcam\r\n",
        "  Returns:\r\n",
        "          img: OpenCV BGR image\r\n",
        "  \"\"\"\r\n",
        "  # decode base64 image\r\n",
        "  image_bytes = b64decode(js_reply.split(',')[1])\r\n",
        "  # convert bytes to numpy array\r\n",
        "  jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)\r\n",
        "  # decode numpy array into OpenCV BGR image\r\n",
        "  img = cv2.imdecode(jpg_as_np, flags=1)\r\n",
        "\r\n",
        "  return img\r\n",
        "\r\n",
        "# function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream\r\n",
        "def bbox_to_bytes(bbox_array):\r\n",
        "  \"\"\"\r\n",
        "  Params:\r\n",
        "          bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.\r\n",
        "  Returns:\r\n",
        "        bytes: Base64 image byte string\r\n",
        "  \"\"\"\r\n",
        "  # convert array into PIL image\r\n",
        "  bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')\r\n",
        "  iobuf = io.BytesIO()\r\n",
        "  # format bbox into png for return\r\n",
        "  bbox_PIL.save(iobuf, format='png')\r\n",
        "  # format return string\r\n",
        "  bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))\r\n",
        "\r\n",
        "  return bbox_bytes"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MaZIOR4WaT64"
      },
      "source": [
        "## Haar Cascade Classifier\r\n",
        "For this tutorial we will run a simple object detection algorithm called Haar Cascade on our images and video fetched from our webcam. OpenCV has a pre-trained Haar Cascade face detection model. "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZpA68lTrcvZs"
      },
      "source": [
        "# initialize the Haar Cascade face detection model\r\n",
        "face_cascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fRFoJo6QT94w"
      },
      "source": [
        "## Webcam Images\r\n",
        "Running code on images taken from webcam is fairly straight-forward. We will utilize code within Google Colab's **Code Snippets** that has a variety of useful code functions to perform various tasks.\r\n",
        "\r\n",
        "We will be using the code snippet for **Camera Capture** to utilize your computer's webcam."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XwUtEojjnPRb"
      },
      "source": [
        "def take_photo(filename='photo.jpg', quality=0.8):\n",
        "  js = Javascript('''\n",
        "    async function takePhoto(quality) {\n",
        "      const div = document.createElement('div');\n",
        "      const capture = document.createElement('button');\n",
        "      capture.textContent = 'Capture';\n",
        "      div.appendChild(capture);\n",
        "\n",
        "      const video = document.createElement('video');\n",
        "      video.style.display = 'block';\n",
        "      const stream = await navigator.mediaDevices.getUserMedia({video: true});\n",
        "\n",
        "      document.body.appendChild(div);\n",
        "      div.appendChild(video);\n",
        "      video.srcObject = stream;\n",
        "      await video.play();\n",
        "\n",
        "      // Resize the output to fit the video element.\n",
        "      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);\n",
        "\n",
        "      // Wait for Capture to be clicked.\n",
        "      await new Promise((resolve) => capture.onclick = resolve);\n",
        "\n",
        "      const canvas = document.createElement('canvas');\n",
        "      canvas.width = video.videoWidth;\n",
        "      canvas.height = video.videoHeight;\n",
        "      canvas.getContext('2d').drawImage(video, 0, 0);\n",
        "      stream.getVideoTracks()[0].stop();\n",
        "      div.remove();\n",
        "      return canvas.toDataURL('image/jpeg', quality);\n",
        "    }\n",
        "    ''')\n",
        "  display(js)\n",
        "\n",
        "  # get photo data\n",
        "  data = eval_js('takePhoto({})'.format(quality))\n",
        "  # get OpenCV format image\n",
        "  img = js_to_image(data) \n",
        "  # grayscale img\n",
        "  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)\n",
        "  print(gray.shape)\n",
        "  # get face bounding box coordinates using Haar Cascade\n",
        "  faces = face_cascade.detectMultiScale(gray)\n",
        "  # draw face bounding box on image\n",
        "  for (x,y,w,h) in faces:\n",
        "      img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)\n",
        "  # save image\n",
        "  cv2.imwrite(filename, img)\n",
        "\n",
        "  return filename"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ilLkpcKanPRb"
      },
      "source": [
        "try:\n",
        "  filename = take_photo('photo.jpg')\n",
        "  print('Saved to {}'.format(filename))\n",
        "  \n",
        "  # Show the image which was just taken.\n",
        "  display(Image(filename))\n",
        "except Exception as err:\n",
        "  # Errors will be thrown if the user does not have a webcam or if they do not\n",
        "  # grant the page permission to access it.\n",
        "  print(str(err))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "I_xcqQZKYzAj"
      },
      "source": [
        "## Webcam Videos\r\n",
        "Running code on webcam video is a little more complex than images. We need to start a video stream using our webcam as input. Then we run each frame through our progam (face detection) and create an overlay image that contains bounding box of detection(s). We then overlay the bounding box image back onto the next frame of our video stream.\r\n",
        "\r\n",
        "![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAGkCAYAAAChGJSeAAAgAElEQVR4AeydB5jURBvHOXqTKlWkF+m9I0V6B5Xey8HBAQeCvfvZEHtBrAiiFAHpIh1pIihItaAg2LHS7058v+fdc5ZsdpJNdtN298/zLMnmJpPJO/+Z+e07LdONU4jwgQ2gAWgAGoAGoAFoABqIHw1kQmbHT2Yjr5HX0AA0AA1AA9AANMAaAADCAwoPMDQADUAD0AA0AA3EmQYAgHGW4fjlh19+0AA0AA1AA9AANAAABADiVx80AA1AA9AANAANxJkGAIBxluH41YdffdAANAANQAPQADQAAAQA4lcfNAANQAPQADQADcSZBgCAcZbh+NWHX33QADQADUAD0AA0AAAEAOJXHzQADUAD0AA0AA3EmQYAgHGW4fjVh1990AA0AA1AA9AANAAABADiVx80AA1AA9AANAANxJkGAIBxluH41YdffdAANAANQAPQADQAAAQA4lcfNAANQAPQADQADcSZBgCAcZbh+NWHX33QADQADUAD0AA0AAAEAOJXHzQADUAD0AA0AA3EmQYAgHGW4fjVh1990AA0AA1AA9AANAAABADiVx80AA1AA9AANAANxJkGAIBxluH41YdffdAANAANQAPQADQAAAQA4lcfNAANQAPQADQADcSZBgCAcZbh+NWHX33QADQADUAD0AA0AAAEAOJXHzQADUAD0AA0AA3EmQYAgHGW4fjVh1990AA0AA1AA9AANAAABADiVx80AA1AA9AANAANxJkGAIBxluH41YdffdAANAANQAPQADQAAAQA4lcfNOBJDfSceMmT6ULDiYYTGoAGYkEDAEA0/mhkbdRA8xs/pGJlOgZ9rr1ukK7dqzS6K+gejqd+x7d89/WefFn695otnwyIVyucSFOJ8j2odLVhVLnBbdS463vUZcxPAfdrVXKV6k+VPl/Eqz7Wbfeabrzdxv1Jtdu8QCUr9qZceUtR5iw5KVOmTJSQORvlzFOcipbpQNWaP0KdRp/UjUcrvbLrHYZ/RWznwiVbUPachSkhIQslZM5KOXIVocIlm1OFuinUrNca6jUpzf/MinUnm3pvpR2uqdzXH88Ng/Zpx1O2M11T6WYqW2M01bh+BrXs+xFxPsreQVyr33GOdnwS/VWoM1E3PtZB/iJ1KH+R2tJP1Sb3697P6dJN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)</center>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ghUlAJzKSjFT"
      },
      "source": [
        "# JavaScript to properly create our live video stream using our webcam as input\r\n",
        "def video_stream():\r\n",
        "  js = Javascript('''\r\n",
        "    var video;\r\n",
        "    var div = null;\r\n",
        "    var stream;\r\n",
        "    var captureCanvas;\r\n",
        "    var imgElement;\r\n",
        "    var labelElement;\r\n",
        "    \r\n",
        "    var pendingResolve = null;\r\n",
        "    var shutdown = false;\r\n",
        "    \r\n",
        "    function removeDom() {\r\n",
        "       stream.getVideoTracks()[0].stop();\r\n",
        "       video.remove();\r\n",
        "       div.remove();\r\n",
        "       video = null;\r\n",
        "       div = null;\r\n",
        "       stream = null;\r\n",
        "       imgElement = null;\r\n",
        "       captureCanvas = null;\r\n",
        "       labelElement = null;\r\n",
        "    }\r\n",
        "    \r\n",
        "    function onAnimationFrame() {\r\n",
        "      if (!shutdown) {\r\n",
        "        window.requestAnimationFrame(onAnimationFrame);\r\n",
        "      }\r\n",
        "      if (pendingResolve) {\r\n",
        "        var result = \"\";\r\n",
        "        if (!shutdown) {\r\n",
        "          captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);\r\n",
        "          result = captureCanvas.toDataURL('image/jpeg', 0.8)\r\n",
        "        }\r\n",
        "        var lp = pendingResolve;\r\n",
        "        pendingResolve = null;\r\n",
        "        lp(result);\r\n",
        "      }\r\n",
        "    }\r\n",
        "    \r\n",
        "    async function createDom() {\r\n",
        "      if (div !== null) {\r\n",
        "        return stream;\r\n",
        "      }\r\n",
        "\r\n",
        "      div = document.createElement('div');\r\n",
        "      div.style.border = '2px solid black';\r\n",
        "      div.style.padding = '3px';\r\n",
        "      div.style.width = '100%';\r\n",
        "      div.style.maxWidth = '600px';\r\n",
        "      document.body.appendChild(div);\r\n",
        "      \r\n",
        "      const modelOut = document.createElement('div');\r\n",
        "      modelOut.innerHTML = \"<span>Status:</span>\";\r\n",
        "      labelElement = document.createElement('span');\r\n",
        "      labelElement.innerText = 'No data';\r\n",
        "      labelElement.style.fontWeight = 'bold';\r\n",
        "      modelOut.appendChild(labelElement);\r\n",
        "      div.appendChild(modelOut);\r\n",
        "           \r\n",
        "      video = document.createElement('video');\r\n",
        "      video.style.display = 'block';\r\n",
        "      video.width = div.clientWidth - 6;\r\n",
        "      video.setAttribute('playsinline', '');\r\n",
        "      video.onclick = () => { shutdown = true; };\r\n",
        "      stream = await navigator.mediaDevices.getUserMedia(\r\n",
        "          {video: { facingMode: \"environment\"}});\r\n",
        "      div.appendChild(video);\r\n",
        "\r\n",
        "      imgElement = document.createElement('img');\r\n",
        "      imgElement.style.position = 'absolute';\r\n",
        "      imgElement.style.zIndex = 1;\r\n",
        "      imgElement.onclick = () => { shutdown = true; };\r\n",
        "      div.appendChild(imgElement);\r\n",
        "      \r\n",
        "      const instruction = document.createElement('div');\r\n",
        "      instruction.innerHTML = \r\n",
        "          '<span style=\"color: red; font-weight: bold;\">' +\r\n",
        "          'When finished, click here or on the video to stop this demo</span>';\r\n",
        "      div.appendChild(instruction);\r\n",
        "      instruction.onclick = () => { shutdown = true; };\r\n",
        "      \r\n",
        "      video.srcObject = stream;\r\n",
        "      await video.play();\r\n",
        "\r\n",
        "      captureCanvas = document.createElement('canvas');\r\n",
        "      captureCanvas.width = 640; //video.videoWidth;\r\n",
        "      captureCanvas.height = 480; //video.videoHeight;\r\n",
        "      window.requestAnimationFrame(onAnimationFrame);\r\n",
        "      \r\n",
        "      return stream;\r\n",
        "    }\r\n",
        "    async function stream_frame(label, imgData) {\r\n",
        "      if (shutdown) {\r\n",
        "        removeDom();\r\n",
        "        shutdown = false;\r\n",
        "        return '';\r\n",
        "      }\r\n",
        "\r\n",
        "      var preCreate = Date.now();\r\n",
        "      stream = await createDom();\r\n",
        "      \r\n",
        "      var preShow = Date.now();\r\n",
        "      if (label != \"\") {\r\n",
        "        labelElement.innerHTML = label;\r\n",
        "      }\r\n",
        "            \r\n",
        "      if (imgData != \"\") {\r\n",
        "        var videoRect = video.getClientRects()[0];\r\n",
        "        imgElement.style.top = videoRect.top + \"px\";\r\n",
        "        imgElement.style.left = videoRect.left + \"px\";\r\n",
        "        imgElement.style.width = videoRect.width + \"px\";\r\n",
        "        imgElement.style.height = videoRect.height + \"px\";\r\n",
        "        imgElement.src = imgData;\r\n",
        "      }\r\n",
        "      \r\n",
        "      var preCapture = Date.now();\r\n",
        "      var result = await new Promise(function(resolve, reject) {\r\n",
        "        pendingResolve = resolve;\r\n",
        "      });\r\n",
        "      shutdown = false;\r\n",
        "      \r\n",
        "      return {'create': preShow - preCreate, \r\n",
        "              'show': preCapture - preShow, \r\n",
        "              'capture': Date.now() - preCapture,\r\n",
        "              'img': result};\r\n",
        "    }\r\n",
        "    ''')\r\n",
        "\r\n",
        "  display(js)\r\n",
        "  \r\n",
        "def video_frame(label, bbox):\r\n",
        "  data = eval_js('stream_frame(\"{}\", \"{}\")'.format(label, bbox))\r\n",
        "  return data"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1nkSnkbkk4cC"
      },
      "source": [
        "# start streaming video from webcam\r\n",
        "video_stream()\r\n",
        "# label for video\r\n",
        "label_html = 'Capturing...'\r\n",
        "# initialze bounding box to empty\r\n",
        "bbox = ''\r\n",
        "count = 0 \r\n",
        "while True:\r\n",
        "    js_reply = video_frame(label_html, bbox)\r\n",
        "    if not js_reply:\r\n",
        "        break\r\n",
        "\r\n",
        "    # convert JS response to OpenCV Image\r\n",
        "    img = js_to_image(js_reply[\"img\"])\r\n",
        "\r\n",
        "    # create transparent overlay for bounding box\r\n",
        "    bbox_array = np.zeros([480,640,4], dtype=np.uint8)\r\n",
        "\r\n",
        "    # grayscale image for face detection\r\n",
        "    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)\r\n",
        "\r\n",
        "    # get face region coordinates\r\n",
        "    faces = face_cascade.detectMultiScale(gray)\r\n",
        "    # get face bounding box for overlay\r\n",
        "    for (x,y,w,h) in faces:\r\n",
        "      bbox_array = cv2.rectangle(bbox_array,(x,y),(x+w,y+h),(255,0,0),2)\r\n",
        "\r\n",
        "    bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255\r\n",
        "    # convert overlay of bbox into bytes\r\n",
        "    bbox_bytes = bbox_to_bytes(bbox_array)\r\n",
        "    # update bbox so next frame gets new overlay\r\n",
        "    bbox = bbox_bytes"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vCBVkX5mNukp"
      },
      "source": [
        "## Hope You Enjoyed!\r\n",
        "If you enjoyed the tutorial and want to see more videos or tutorials check out my YouTube channel [HERE](https://www.youtube.com/channel/UCrydcKaojc44XnuXrfhlV8Q?sub_confirmation=1)\r\n",
        "\r\n",
        "Have a great day!"
      ]
    }
  ]
}