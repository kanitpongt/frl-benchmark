from sre_constants import SUCCESS
import time
from enum import Enum
from typing import Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from deepface import DeepFace as df

class VerifyLibrary(str, Enum):
    deepface = "DeepFace"
    insight_face = "InsightFace"
    
class DeepFaceRecognitionModel(str, Enum):
    vgg_face = "VGG-Face"
    facenet = "Facenet"
    facenet512 = "Facenet512"
    openface = "OpenFace"
    deepface = "DeepFace"
    deepId = "DeepID"
    arcface = "ArcFace"
    dlib = "Dlib"

class DeepFaceDetectorModel(str, Enum):
    cv2 = "opencv"
    dlib = "dlib"
    ssd = "ssd"
    retinaface = "retinaface"
    mtcnn = "mtcnn"


class DeepFaceMetric(str, Enum):
    cos = "cosine"
    euclid = "euclidean"
    euclid_norm = "euclidean_l2"

class ImagePair(BaseModel):
    img1: str
    img2: str
    
class IndexResponse(BaseModel):
    available_library: List[VerifyLibrary]
class VerifyResponse(BaseModel):
    verified: bool
    distance: float
    max_threshold_to_verify: float
    model: DeepFaceRecognitionModel
    detector_model: DeepFaceDetectorModel
    similarity_metric: DeepFaceMetric
    seconds: float
    success: bool
    
class ErrorResponse(BaseModel):
    success: bool
    error: str


class VerifyRequest(BaseModel):
    images: List[ImagePair]
    library: VerifyLibrary

class DeepFaceVerifyRequest(VerifyRequest):
    detector_backend: DeepFaceDetectorModel = DeepFaceDetectorModel.cv2
    model_name: DeepFaceRecognitionModel = DeepFaceRecognitionModel.vgg_face
    distance_metric: DeepFaceMetric = DeepFaceMetric.cos

DEFAULT_LIBRARY: VerifyLibrary = "deepface"

router = APIRouter()

@router.get('/', response_model=IndexResponse, status_code=status.HTTP_200_OK)
def verify_face():
    return {"available_library": list(VerifyLibrary)}

@router.post('/deepface', response_model=VerifyResponse)
def deepface_verify(req: DeepFaceVerifyRequest):
    
    # Get model, detector model, and similarity_metric
    model_name = req.model_name
    distance_metric = req.distance_metric
    detector_backend = req.detector_backend
    
    # Get and verify image pair passed in request
    instances = req.images[0]
    check_image_pair(instances)
    
    # Run deepface verify function and time it
    tic = time.time()
    resp_obj = deepface_verify_wrapper(image=instances, model= model_name, detector=detector_backend, metric=distance_metric)
    toc = time.time()
    
    # If no error, add time and model used to response
    if resp_obj != None and resp_obj["success"]:
        resp_obj["detector_model"] = detector_backend
        resp_obj["seconds"] = toc-tic
    
    # Return response object
    return resp_obj

def deepface_verify_wrapper(image: ImagePair, model: DeepFaceDetectorModel, detector: DeepFaceDetectorModel, metric: DeepFaceMetric):
    try:
        resp_obj = df.verify(image.img1, image.img2, model_name=model, distance_metric=metric, detector_backend=detector)
        resp_obj["success"] = True
    except Exception as err:
        resp_obj = {'success': False, 'error': str(err)}
    return resp_obj

def insight_verify(images, model, distance, detector):
    return "TODO"

def check_image_pair(image_pair: ImagePair):
    validate_img1 = False
    if len(image_pair.img1) > 11 and image_pair.img1[0:11] == "data:image/":
        validate_img1 = True
    validate_img2 = False
    if len(image_pair.img2) > 11 and image_pair.img2[0:11] == "data:image/":
        validate_img2 = True
    if validate_img1 != True or validate_img2 != True:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='you must pass both image as base64 encoded string')