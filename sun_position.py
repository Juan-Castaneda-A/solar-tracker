import numpy as np
from math import *
import calendar

def doy_tod_conv(date_and_time,longitude,time_zone_center_longitude):
    """
    Toma un único objeto datetime.datetime como entrada. 
    Retorna dos valores: el primero es el día del año
    y el segundo es la hora del día en segundos (reloj de 24 horas).
    """
    # Corrección del tiempo. El centro del horario PST está en -120 W, mientras que la longitud del lugar de interés está aprox. 2° al oeste del centro del PST
    pst_center_longitude = time_zone_center_longitude  # el signo negativo indica longitud oeste
    loc_longitude = longitude  # el signo negativo indica longitud oeste
    correction = np.abs(60/15*(loc_longitude - pst_center_longitude))
    min_correction = int(correction)  # retraso horario local en minutos respecto al PST
    sec_correction = int((correction - min_correction)*60)  # retraso horario local en segundos respecto al PST

    # Ajuste de hora con base en la corrección
    if date_and_time.minute <= min_correction:
        date_and_time = date_and_time.replace(hour=date_and_time.hour-1, minute=60 + date_and_time.minute - min_correction - 1, second=60 - sec_correction)
    else:
        date_and_time = date_and_time.replace(minute=date_and_time.minute - min_correction - 1, second=60 - sec_correction)

    time_of_day = date_and_time.hour * 3600 + date_and_time.minute * 60 + date_and_time.second

    # El siguiente fragmento de código calcula el día del año
    months = [31,28,31,30,31,30,31,31,30,31,30,31]  # días en cada mes
    if (date_and_time.year % 4 == 0) and (date_and_time.year % 100 != 0 or date_and_time.year % 400 == 0) == True:
        months[1] = 29  # Modificación para años bisiestos
    day_of_year = sum(months[:date_and_time.month - 1]) + date_and_time.day

    # Corrección por horario de verano (NOTA: No funciona para la primera hora de cada día durante el periodo de horario de verano)
    # qué día del año es el segundo domingo de marzo ese año
    dst_start_day = sum(months[:2]) + calendar.monthcalendar(date_and_time.year, date_and_time.month)[1][6]
    # qué día del año es el primer domingo de noviembre ese año
    dst_end_day = sum(months[:10]) + calendar.monthcalendar(date_and_time.year, date_and_time.month)[0][6]
    if day_of_year >= dst_start_day and day_of_year < dst_end_day:
        time_of_day = time_of_day - 3600

    return day_of_year, time_of_day

def solar_angle(times, latitude=37.424107, longitude=-122.174199, time_zone_center_longitude=-120):
    """
    Calcula los ángulos solares (Acimut, Cenital) para una ubicación específica.
    Entrada: marca de tiempo en formato datetime.datetime,
    latitud y longitud de la ubicación en grados
    time_zone_center_longitude (para corrección horaria local): longitud en grados del centro del huso horario (ej. para PST es -120)
    """

    day_of_year, time_of_day = doy_tod_conv(times, longitude, time_zone_center_longitude)
    latitude = radians(latitude)  # Coordenada latitudinal de Stanford

    # Cálculo de parámetros dependientes del tiempo, día y ubicación. Referencia: libro de texto de DaRosa
    alpha = 2*pi*(time_of_day - 43200) / 86400  # Ángulo horario en radianes
    delta = radians(23.44 * sin(radians((360 / 365.25) * (day_of_year - 80))))  # Ángulo de declinación solar
    chi = acos(sin(delta)*sin(latitude) + cos(delta)*cos(latitude)*cos(alpha))  # Ángulo cenital del sol
    tan_xi = sin(alpha) / (sin(latitude)*cos(alpha) - cos(latitude)*tan(delta))  # tangente del ángulo acimutal del sol, xi

    # Conversión de tangente a ángulo en radianes con condiciones por cuadrante
    if alpha > 0 and tan_xi > 0:
        xi = pi + atan(tan_xi)
    elif alpha > 0 and tan_xi < 0:
        xi = 2*pi + atan(tan_xi)
    elif alpha < 0 and tan_xi > 0:
        xi = atan(tan_xi)
    else:
        xi = pi + atan(tan_xi)

    return degrees(xi), degrees(chi)

# <--- MODIFICACIÓN PRINCIPAL AQUÍ --->
def sun_position(time, latitude, longitude, time_zone_center_longitude):
    """
    Toma la marca de tiempo, latitud, longitud y huso horario.
    Devuelve la posición del sol (x, y) en coordenadas cartesianas, y una máscara binaria del sol.
    """

    # Parámetros por defecto
    delta = 14.036  # diferencia entre el norte geográfico y el norte en la imagen del cielo
    r = 29  # radio de la imagen del cielo (el círculo)
    origin_x = 29  # Coordenada x del centro de la imagen
    origin_y = 30  # Coordenada y del centro de la imagen

    # <--- MODIFICACIÓN PRINCIPAL AQUÍ --->
    # Ahora pasamos los parámetros de ubicación a la función solar_angle
    azimuth, zenith = solar_angle(time, latitude, longitude, time_zone_center_longitude)
    
    rho = zenith / 90 * r  # coordenada polar - distancia
    theta = azimuth - delta + 90  # coordenada polar - ángulo
    sun_center_x = round(origin_x - rho * sin(radians(theta)))
    sun_center_y = round(origin_y + rho * cos(radians(theta)))

    # Generación de la máscara solar
    sun_mask = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            if (i - sun_center_x)**2 + (j - sun_center_y)**2 <= 2**2:
                sun_mask[:, :, 0][i, j] = 255

    return sun_center_x, sun_center_y, sun_mask
