import numpy as np

# Imágenes de pruebas
from skimage import data, segmentation, color, feature, filters

from skimage.segmentation import felzenszwalb, quickshift, watershed, mark_boundaries
from skimage.measure import regionprops
from skimage.exposure import rescale_intensity

# Watershed con marcadores
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
from skimage.filters import rank
from skimage.morphology import disk, diamond
from skimage.color import rgb2gray

# Region Adjancency Graphs (RAGs)
from skimage.future import graph

# Para lectura del dataset de imágenes segmentadas
import pandas as pd

import matplotlib.pyplot as plt

def calculate_metrics(labeled_img, gt):
    """
    Función para calcular las métricas de segmentación
    
    Args:
        labeled_img: Numpy Array
            Imagen binarizada de la segmentación calculada 
        gt: Numpy Array
            Ground True
            
    Return:
        (accuracy, error_rate, precission, recall, F1, sensibility, specificity, iou)
    """
    # Comparación de pixeles para obtener las siguientes metricas
    # TP pixel >0 en imagen segmentada == pixel > 0 en imagen de mascara de la muestra
    # TN pixel = 0 en imagen segmentada == pixel = 0 en imagen de mascara de la muestra
    # FP pixel >0 en imagen segmentada pero debería ser 0
    # FN pixel = 0 en imagen segmentada pero debería ser >0

    # Inicializamos valores de metricas
    TP=TN=FP=FN = 0

    # Obtenemos dimensiones de la imagen
    img_shape = gt.shape

    num_predict= 0
    num_origin = 0

    # Minimos para considerar una etiqueta fg
    th_gt = np.min(gt)
    th_lb = np.min(labeled_img)


    # Recorremos los pixeles para comparar valores
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if labeled_img[i][j] > th_lb and gt[i][j] > th_gt:
                TP += 1
                num_predict += 1
                num_origin += 1
            elif labeled_img[i][j] <= th_lb and gt[i][j] <= th_gt:
                TN += 1
            elif labeled_img[i][j] > th_lb and gt[i][j] <= th_gt:
                FP += 1
                num_predict += 1
            elif labeled_img[i][j] <= th_lb and gt[i][j] > th_gt:
                FN += 1
                num_origin += 1

    accuracy = (TP + TN)/(TP + TN + FP + FN)
    error_rate = 1-accuracy
    precission = TP/(TP + FP)
    recall = TP/(TP + FN)
    F1 = (2*precission*recall)/(precission + recall)
    sensibility = recall
    specificity = TN/(TN + FP)

    intersection = np.logical_and(rgb2gray(labeled_img), gt)
    union = np.logical_or(rgb2gray(labeled_img), gt)
    iou = np.sum(intersection) / np.sum(union)
    
    return (accuracy, error_rate, precission, recall, F1, sensibility, specificity, iou, num_predict, num_origin)

def get_reference_masks(path_dataframe, imageIds):
    """Función para obtener un array de máscaras de referencia de 
       las imágenes a analizar

    Args:
        path_dataframe (String): Path del fichero csv que contiene los 
                                  datos para cargar en el objeto data frame
                                  de pandas
        imageIds ([String]): Array de nombres de las imágenes a analizar

    Returns:
        [Numpy Arrays]: Array de imágenes de las máscaras de referencia
    """

    df_masks = pd.read_csv(path_dataframe)
    list_masks = []
    for image_name in imageIds:
        image_rgb = plt.imread(image_name)

        df_rows = df_masks.loc[df_masks['ImageId'] == image_name]

        mask_img = np.zeros((image_rgb.shape[0]*image_rgb.shape[1],1), dtype=int)
        l,b=image_rgb.shape[0], image_rgb.shape[1]

        for row in df_rows.iterrows():
            en_pix = row[1]['EncodedPixels']

            rle = list(map(int, en_pix.split(' ')))
            pixel,pixel_count = [],[]
            [pixel.append(rle[i]) if i%2==0 else pixel_count.append(rle[i]) for i in range(0, len(rle))]

            rle_pixels = [list(range(pixel[i],pixel[i]+pixel_count[i])) for i in range(0, len(pixel))]

            rle_mask_pixels = sum(rle_pixels,[]) 

            mask_img[rle_mask_pixels] = 255
            
            mask = np.reshape(mask_img, (b,l)).T

        list_masks.append(mask)

    return list_masks


def segm_felzenszwalb(input_image, scale=800, sigma=10,
                      min_size=5, multichannel=True):
    """Función para generar la segmentación mediante la técnica
    de Felzenszwalb.

    Args:
        input_image ([Numpy Array]): Imagen de entrada sobre la que obtener
                                    la segmentación
        scale (int, optional): A mayor, más grandes son los grupos.
                               Defaults to 800.
        sigma (int, optional): Desviación estándar del kernel Gaussiano.
                                Defaults to 10.
        min_size (int, optional): Tamano mínimo de los componentes.
                                  Defaults to 5.
        multichannel (bool, optional): Si el último valor del shape
                                       se interpreta como múltiples canales.
                                       Defaults to True.
    Returns:
        Tuple (output image, labels, number classes): Tupla con la imagen
                                            segmentada, las etiquetas y el número
                                            total de segmentos encontrados.
    """

    segments_felz = felzenszwalb(np.uint8(input_image),
                          scale=scale,
                          sigma=sigma,
                          min_size=min_size,
                          multichannel=multichannel)

    output_image = mark_boundaries(input_image, segments_felz)
    labeled_fz = color.label2rgb(segments_felz, input_image, kind='avg', bg_label=0)

    return (output_image, labeled_fz, len(np.unique(segments_felz)))

def segm_watershed(input_image, gradient_level=10,
                   denoised_d_radius=10, rank_d_radius=2,
                   compactness=1):
    """Función para generar la segmentación mediante la técnica
    de WaterShed.

    Args:
        input_image ([Numpy Array]): Imagen de entrada sobre la que obtener
                                    la segmentación.
        gradient_level (int, optional): Nivel del gradiente para encontrar regiones
                                        continuas. Defaults to 10.
        denoised_d_radius (int, optional): Radio del elemento morfologico 'diamond'
                                           para obtener una imagen más suave.Defaults to 10.
        rank_d_radius (int, optional): Radio del elemento morfológico 'diamond'
                                       para expresar la vecindad. Defaults to 2.
        compactness (int, optional): A mayor valor, se dan lugar cuencas de forma más regular.
                                     Defaults to 1.

    Returns:
        Tuple (output image, labels, number classes): Tupla con la imagen
                                            segmentada, las etiquetas y el número
                                            total de segmentos encontrados.
    """
    input_image = rgb2gray(input_image)
    # denoise image
    denoised = rank.median(input_image, diamond(denoised_d_radius))

    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(denoised, diamond(rank_d_radius)) < gradient_level
    markers = ndi.label(markers)[0]

    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(denoised, diamond(5))

    # process the watershed
    segments_watershed = watershed(gradient, markers,
                    compactness=compactness)

    output_image = mark_boundaries(input_image, segments_watershed)
    labeled_ws = color.label2rgb(segments_watershed, input_image, kind='avg', bg_label=0)

    return (output_image, labeled_ws, len(np.unique(segments_watershed)))


def segm_quickshift(input_image, ratio=0.1, kernel_size=5,
                    max_dist=25, sigma=0.8):
    """Función para generar la segmentación mediante la técnica
    de Quickshift.

    Args:
        input_image ([Numpy Array]): Imagen de entrada sobre la que obtener
                                    la segmentación.
        ratio (float, optional): Balance del espacio de color.
                                 Defaults to 0.1.
        kernel_size (int, optional): Ancho del kernel Gaussiano. A mayor valor,
                                     menos clusters. Defaults to 5.
        max_dist (int, optional): Punto de corte para las distancias de los datos.
                                  Más alto significa menos clusters. Defaults to 25.
        sigma (float, optional): Anchura para el suavizado Gaussiano. Defaults to 0.8.

    Returns:
         Tuple (output image, labels, number classes): Tupla con la imagen
                                            segmentada, las etiquetas y el número
                                            total de segmentos encontrados.
    """

    segments_quick = quickshift(input_image,
                        ratio=ratio,
                        kernel_size=kernel_size,
                        max_dist=max_dist,
                        sigma=sigma)
    output_image = mark_boundaries(input_image, segments_quick)
    labeled_qs = color.label2rgb(segments_quick, input_image, kind='avg', bg_label=0)

    return (output_image, labeled_qs, len(np.unique(segments_quick)))


def segm_rag(input_image, compactness=10, n_segments=275,
             sigma=1, start_label=1):
    """Función para generar la segmentación mediante la técnica
    de RAG.

    Args:
        input_image ([Numpy Array]): Imagen de entrada sobre la que obtener
                                    la segmentación.
        compactness (int, optional): Equilibra proximidad de color y espacio.
                                     Defaults to 10.
        n_segments (int, optional): El número aproximado de etiquetas en la imagen
                                    de salida. Defaults to 275.
        sigma (int, optional): Anchura del núcleo de suavizado Gaussiano. Defaults to 1.
        start_label (int, optional): Índice inicial de etiquetado 0/1. Defaults to 1.

    Returns:
        Tuple (output image, labels, number classes): Tupla con la imagen
                                            segmentada, las etiquetas y el número
                                            total de segmentos encontrados.
    """

    labels = segmentation.slic(input_image, 
                           compactness=compactness,
                           n_segments=n_segments,
                           sigma=sigma,
                           start_label=start_label)
    g = graph.rag_mean_color(input_image, labels)

    segments_rag = graph.cut_threshold(labels, 
                                g,
                                29)

    output_image = mark_boundaries(input_image, segments_rag)#
    labeled_rag = color.label2rgb(segments_rag, input_image, kind='avg', bg_label=0)

    return (output_image, labeled_rag, len(np.unique(segments_rag)))