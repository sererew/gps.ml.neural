#!/usr/bin/env python3
"""
Script de prueba para validar las funciones de conversiÃ³n de coordenadas
y cÃ¡lculo/integraciÃ³n de deltas del filtro neuronal.
"""

import numpy as np
import pandas as pd
import sys
import os

# Anadir python/filters al path para importar las funciones
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'filters'))

from importlib import import_module
nn_filter = import_module('7_nn_filter')

def test_coordinate_conversion():
    """Prueba la conversiÃ³n lat/lon <-> metros."""
    print("=== TEST: ConversiÃ³n de coordenadas ===")
    
    # Coordenadas de prueba (Madrid aproximadamente)
    lat_ref, lon_ref = 40.4168, -3.7038
    
    # Puntos de prueba
    test_points = [
        (40.4168, -3.7038),    # Punto de referencia (debe ser 0,0)
        (40.4178, -3.7038),    # 1000m al norte aprox
        (40.4168, -3.6938),    # 1000m al este aprox 
        (40.4158, -3.7138),    # 1000m al sur y oeste aprox
    ]
    
    print(f"Punto de referencia: lat={lat_ref}, lon={lon_ref}")
    print()
    
    for i, (lat, lon) in enumerate(test_points):
        # Convertir a metros
        x, y = nn_filter.latlon_to_meters(np.array([lat]), np.array([lon]), lat_ref, lon_ref)
        
        # Convertir de vuelta a lat/lon
        lat_back, lon_back = nn_filter.meters_to_latlon(x, y, lat_ref, lon_ref)
        
        # Calcular errores
        lat_error = abs(lat - lat_back[0]) * 111320  # metros
        lon_error = abs(lon - lon_back[0]) * 111320 * np.cos(np.radians(lat_ref))  # metros
        
        print(f"Punto {i+1}: lat={lat:.6f}, lon={lon:.6f}")
        print(f"  -> metros: x={x[0]:.2f}, y={y[0]:.2f}")
        print(f"  -> vuelta: lat={lat_back[0]:.6f}, lon={lon_back[0]:.6f}")
        print(f"  -> error: {lat_error:.3f}m lat, {lon_error:.3f}m lon")
        print()

def test_delta_integration():
    """Prueba el cÃ¡lculo e integraciÃ³n de deltas."""
    print("=== TEST: CÃ¡lculo e integraciÃ³n de deltas ===")
    
    # Coordenadas de prueba: cuadrado de 100m x 100m
    x_orig = np.array([0, 100, 100, 0, 0])  # cuadrado
    y_orig = np.array([0, 0, 100, 100, 0])
    z_orig = np.array([0, 10, 20, 10, 0])
    
    print("Coordenadas originales:")
    for i in range(len(x_orig)):
        print(f"  P{i}: x={x_orig[i]:6.1f}, y={y_orig[i]:6.1f}, z={z_orig[i]:6.1f}")
    print()
    
    # Calcular deltas
    dx, dy, dz = nn_filter.calculate_deltas(x_orig, y_orig, z_orig)
    
    print("Deltas calculados:")
    for i in range(len(dx)):
        print(f"  Î”P{i}: dx={dx[i]:6.1f}, dy={dy[i]:6.1f}, dz={dz[i]:6.1f}")
    print()
    
    # Integrar deltas para recuperar coordenadas
    x_recov, y_recov, z_recov = nn_filter.integrate_deltas(dx, dy, dz, x_orig[0], y_orig[0], z_orig[0])
    
    print("Coordenadas recuperadas:")
    for i in range(len(x_recov)):
        print(f"  P{i}: x={x_recov[i]:6.1f}, y={y_recov[i]:6.1f}, z={z_recov[i]:6.1f}")
    print()
    
    # Calcular errores
    x_error = np.abs(x_orig - x_recov)
    y_error = np.abs(y_orig - y_recov) 
    z_error = np.abs(z_orig - z_recov)
    
    print("Errores de recuperaciÃ³n:")
    for i in range(len(x_error)):
        print(f"  E{i}: x={x_error[i]:6.3f}, y={y_error[i]:6.3f}, z={z_error[i]:6.3f}")
    print()
    
    max_error = max(np.max(x_error), np.max(y_error), np.max(z_error))
    print(f"Error mÃ¡ximo: {max_error:.6f} metros")
    
    if max_error < 1e-10:
        print("âœ… IntegraciÃ³n de deltas: CORRECTA")
    else:
        print("âŒ IntegraciÃ³n de deltas: ERROR")
    
    return max_error < 1e-10

def test_full_pipeline():
    """Prueba el pipeline completo: lat/lon -> deltas -> filtro simulado -> vuelta."""
    print("=== TEST: Pipeline completo ===")
    
    # Track de prueba: lÃ­nea recta de 1km
    lat_ref, lon_ref = 40.4168, -3.7038
    n_points = 100
    
    # Crear track sintÃ©tico (lÃ­nea recta hacia el norte)
    lat_track = np.linspace(lat_ref, lat_ref + 0.01, n_points)  # ~1km al norte
    lon_track = np.full(n_points, lon_ref)
    ele_track = np.linspace(100, 200, n_points)  # subida gradual
    
    print(f"Track de prueba: {n_points} puntos")
    print(f"  Inicio: lat={lat_track[0]:.6f}, lon={lon_track[0]:.6f}, ele={ele_track[0]:.1f}")
    print(f"  Final:  lat={lat_track[-1]:.6f}, lon={lon_track[-1]:.6f}, ele={ele_track[-1]:.1f}")
    print()
    
    # Paso 1: Convertir a metros
    x, y = nn_filter.latlon_to_meters(lat_track, lon_track, lat_ref, lon_ref)
    z = ele_track
    
    # Paso 2: Calcular deltas
    dx, dy, dz = nn_filter.calculate_deltas(x, y, z)
    
    # Paso 3: Simular filtro (sin cambios para esta prueba)
    dx_filtered = dx.copy()
    dy_filtered = dy.copy()
    dz_filtered = dz.copy()
    
    # Paso 4: Integrar deltas filtrados
    x_filt, y_filt, z_filt = nn_filter.integrate_deltas(dx_filtered, dy_filtered, dz_filtered, x[0], y[0], z[0])
    
    # Paso 5: Convertir de vuelta a lat/lon
    lat_filt, lon_filt = nn_filter.meters_to_latlon(x_filt, y_filt, lat_ref, lon_ref)
    
    # Calcular errores
    lat_errors = np.abs(lat_track - lat_filt) * 111320  # metros
    lon_errors = np.abs(lon_track - lon_filt) * 111320 * np.cos(np.radians(lat_ref))
    ele_errors = np.abs(ele_track - z_filt)
    
    print("Errores en el pipeline completo:")
    print(f"  Latitud: max={np.max(lat_errors):.3f}m, mean={np.mean(lat_errors):.6f}m")
    print(f"  Longitud: max={np.max(lon_errors):.3f}m, mean={np.mean(lon_errors):.6f}m") 
    print(f"  ElevaciÃ³n: max={np.max(ele_errors):.3f}m, mean={np.mean(ele_errors):.6f}m")
    print()
    
    total_error = np.max([np.max(lat_errors), np.max(lon_errors), np.max(ele_errors)])
    
    if total_error < 0.01:  # Error menor a 1cm
        print("âœ… Pipeline completo: CORRECTO")
        return True
    else:
        print("âŒ Pipeline completo: ERROR")
        return False

if __name__ == "__main__":
    print("Ejecutando tests de validaciÃ³n del filtro neuronal...")
    print("=" * 60)
    print()
    
    try:
        test_coordinate_conversion()
        print()
        
        delta_ok = test_delta_integration()
        print()
        
        pipeline_ok = test_full_pipeline()
        print()
        
        if delta_ok and pipeline_ok:
            print("ðŸŽ‰ Todos los tests PASARON. El filtro deberÃ­a funcionar correctamente.")
        else:
            print("âš ï¸  Algunos tests FALLARON. Revisar las funciones.")
            
    except Exception as e:
        print(f"âŒ Error ejecutando tests: {e}")
        import traceback
        traceback.print_exc()

