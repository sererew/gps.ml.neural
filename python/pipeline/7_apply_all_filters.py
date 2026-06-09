#!/usr/bin/env python3
"""
Script maestro para aplicar todos los filtros a todas las pasadas.

Este script:
1. Busca todos los tracks procesados en data/preprocessed/<pasada>/
2. Aplica todos los filtros disponibles a cada track
3. Guarda los resultados en results/filtered/<filtro>/<pasada>/

Filtros implementados:
- nn (red neuronal)
- identity (sin filtrado)
- moving_average
- triangular_weighted  
- median
- savgol
- exponential
- gaussian
- kalman

Uso:
    python 7_apply_all_filters.py
    python 7_apply_all_filters.py --pasadas 1,2,3
    python 7_apply_all_filters.py --filtros nn,kalman,savgol
    python 7_apply_all_filters.py --overwrite
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import glob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from datetime import datetime

def find_track_files(preprocessed_dir):
    """
    Encuentra todos los archivos de tracks procesados.
    
    Returns:
        dict: {pasada: [track_files]}
    """
    tracks_by_pasada = {}
    
    # Buscar todas las carpetas de pasadas
    pasadas_dirs = glob.glob(os.path.join(preprocessed_dir, "*"))
    pasadas_dirs = [d for d in pasadas_dirs if os.path.isdir(d)]
    
    for pasada_dir in pasadas_dirs:
        pasada = os.path.basename(pasada_dir)
        
        # Buscar archivos GPX resampleados (excluyendo patrones)
        gpx_files = glob.glob(os.path.join(pasada_dir, "*_resampled.gpx"))
        
        # Filtrar archivos que NO sean patrones
        track_files = []
        for gpx_file in gpx_files:
            filename = os.path.basename(gpx_file)
            # Excluir archivos de patron
            if not ("pattern" in filename or "aligned_pattern" in filename):
                track_files.append(gpx_file)
        
        if track_files:
            tracks_by_pasada[pasada] = sorted(track_files)
    
    return tracks_by_pasada

def get_available_filters():
    """
    Obtiene la lista de filtros disponibles basandose en python/filters/7_*.py.

    Returns:
        dict: {filter_name: script_path}
    """
    filters = {}
    script_dir = Path(__file__).resolve().parent
    filters_dir = script_dir.parent / "filters"

    # Buscar todos los scripts de filtros en la carpeta organizada.
    filter_scripts = sorted(filters_dir.glob("7_*_filter.py"))

    for script in filter_scripts:
        # Extraer nombre del filtro del nombre del script.
        # Ejemplo: "7_nn_filter.py" -> "nn"
        script_name = script.name
        if script_name.startswith("7_") and script_name.endswith("_filter.py"):
            filter_name = script_name[2:-10]  # Quitar "7_" al inicio y "_filter.py" al final
            filters[filter_name] = str(script)

    return filters

def create_output_path(filter_name, pasada, input_file):
    """
    Genera la ruta de salida para un track filtrado.
    
    Args:
        filter_name: Nombre del filtro (ej: "nn", "kalman")
        pasada: Numero de pasada
        input_file: Archivo de entrada
        
    Returns:
        str: Ruta completa de salida
    """
    input_filename = os.path.basename(input_file)
    name_without_ext = os.path.splitext(input_filename)[0]
    
    # Crear nombre de salida: <original>_<filtro>_filtered.gpx
    output_filename = f"{name_without_ext}_{filter_name}_filtered.gpx"
    
    # Output path: results/filtered/<filter>/<pass>/
    output_dir = os.path.join("results", "filtered", filter_name, pasada)
    output_path = os.path.join(output_dir, output_filename)
    
    return output_path

def run_filter(filter_script, input_file, output_file):
    """
    Ejecuta un filtro especifico en un archivo.
    
    Args:
        filter_script: Script del filtro a ejecutar
        input_file: Archivo de entrada
        output_file: Archivo de salida
        
    Returns:
        tuple: (success, filter_name, input_file, output_file, message)
    """
    filter_name = os.path.basename(filter_script)[2:-10]  # Extraer nombre del filtro
    
    try:
        # Crear directorio de salida
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Ejecutar el filtro
        cmd = [sys.executable, filter_script, input_file, output_file]
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutos maximo por filtro
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            return (True, filter_name, input_file, output_file, f"Success in {elapsed:.1f}s")
        else:
            return (False, filter_name, input_file, output_file, f"Error: {result.stderr.strip()}")
            
    except subprocess.TimeoutExpired:
        return (False, filter_name, input_file, output_file, "Timeout (>5min)")
    except Exception as e:
        return (False, filter_name, input_file, output_file, f"Exception: {str(e)}")

def filter_task(args):
    """Wrapper para ejecutar filtros en paralelo."""
    return run_filter(*args)

def apply_filters_to_tracks(tracks_by_pasada, available_filters, selected_filters=None, selected_pasadas=None, overwrite=False, max_workers=4):
    """
    Aplica los filtros seleccionados a los tracks de las pasadas seleccionadas.
    
    Args:
        tracks_by_pasada: Dict con tracks por pasada
        available_filters: Dict con filtros disponibles
        selected_filters: Lista de filtros a aplicar (None = todos)
        selected_pasadas: Lista de pasadas a procesar (None = todas)
        overwrite: Si sobrescribir archivos existentes
        max_workers: Numero maximo de procesos paralelos
    """
    # Filtrar pasadas si se especifican
    if selected_pasadas is not None:
        tracks_by_pasada = {p: tracks for p, tracks in tracks_by_pasada.items() if p in selected_pasadas}
    
    # Filtrar filtros si se especifican
    if selected_filters is not None:
        available_filters = {f: script for f, script in available_filters.items() if f in selected_filters}
    
    # Crear lista de tareas
    tasks = []
    total_tracks = 0
    
    for pasada, track_files in tracks_by_pasada.items():
        for track_file in track_files:
            for filter_name, filter_script in available_filters.items():
                output_path = create_output_path(filter_name, pasada, track_file)
                
                # Verificar si ya existe y no se debe sobrescribir
                if os.path.exists(output_path) and not overwrite:
                    continue
                
                tasks.append((filter_script, track_file, output_path))
                total_tracks += 1
    
    if not tasks:
        print("No hay tareas que procesar (todos los archivos ya existen y no se especifico --overwrite)")
        return
    
    print(f"\n[INICIO] Iniciando procesamiento de {total_tracks} combinaciones filtro-track")
    print(f"   Pasadas: {list(tracks_by_pasada.keys())}")
    print(f"   Filtros: {list(available_filters.keys())}")
    print(f"   Procesos paralelos: {max_workers}")
    print(f"   Sobrescribir: {'Si' if overwrite else 'No'}")
    print()
    
    # Ejecutar tareas en paralelo
    start_time = time.time()
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Enviar todas las tareas
        futures = [executor.submit(filter_task, task) for task in tasks]
        
        # Procesar resultados conforme van completandose
        for i, future in enumerate(futures):
            try:
                success, filter_name, input_file, output_file, message = future.result()
                
                if success:
                    successful += 1
                    status = "[OK]"
                else:
                    failed += 1
                    status = "[ERROR]"
                
                # Mostrar progreso
                progress = (i + 1) / len(futures) * 100
                input_basename = os.path.basename(input_file)
                print(f"{status} [{progress:5.1f}%] {filter_name:15} {input_basename:40} - {message}")
                
            except Exception as e:
                failed += 1
                print(f"[ERROR] [Error] Tarea fallo: {str(e)}")
    
    elapsed = time.time() - start_time
    
    print(f"\n[RESUMEN] Resumen del procesamiento:")
    print(f"   Tiempo total: {elapsed:.1f} segundos")
    print(f"   Exitosos: {successful}")
    print(f"   Fallidos: {failed}")
    print(f"   Total: {successful + failed}")
    
    if failed > 0:
        print(f"\n[WARNING] {failed} tareas fallaron. Revisar los mensajes de error arriba.")

def main():
    """Funcion principal."""
    parser = argparse.ArgumentParser(description='Apply all filters to all track passes')
    parser.add_argument('--pasadas', type=str, help='Comma-separated list of passes to process (e.g., "1,2,3")')
    parser.add_argument('--filtros', type=str, help='Comma-separated list of filters to apply (e.g., "nn,kalman")')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing filtered files')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of parallel processes (default: 4)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be processed without actually doing it')
    
    args = parser.parse_args()
    
    try:
        print("[INFO] Buscando tracks preprocessados...")
        
        # Buscar todos los tracks
        preprocessed_dir = os.path.join("data", "preprocessed")
        if not os.path.exists(preprocessed_dir):
            print(f"[ERROR] Error: Directorio {preprocessed_dir} no existe")
            sys.exit(1)
        
        tracks_by_pasada = find_track_files(preprocessed_dir)
        
        if not tracks_by_pasada:
            print("[ERROR] No se encontraron tracks preprocessados")
            sys.exit(1)
        
        total_tracks = sum(len(files) for files in tracks_by_pasada.values())
        print(f"   Encontradas {len(tracks_by_pasada)} pasadas con {total_tracks} tracks totales")
        
        # Obtener filtros disponibles
        print("\n[INFO] Buscando filtros disponibles...")
        available_filters = get_available_filters()
        
        if not available_filters:
            print("[ERROR] No se encontraron scripts de filtros en python/filters (7_*_filter.py)")
            sys.exit(1)
        
        print(f"   Filtros disponibles: {list(available_filters.keys())}")
        
        # Parsear argumentos
        selected_pasadas = None
        if args.pasadas:
            selected_pasadas = [p.strip() for p in args.pasadas.split(',')]
            invalid_pasadas = [p for p in selected_pasadas if p not in tracks_by_pasada]
            if invalid_pasadas:
                print(f"[ERROR] Pasadas invalidas: {invalid_pasadas}")
                print(f"   Pasadas disponibles: {list(tracks_by_pasada.keys())}")
                sys.exit(1)
        
        selected_filters = None
        if args.filtros:
            selected_filters = [f.strip() for f in args.filtros.split(',')]
            invalid_filters = [f for f in selected_filters if f not in available_filters]
            if invalid_filters:
                print(f"[ERROR] Filtros invalidos: {invalid_filters}")
                print(f"   Filtros disponibles: {list(available_filters.keys())}")
                sys.exit(1)
        
        # Mostrar resumen de lo que se va a procesar
        pasadas_to_process = selected_pasadas if selected_pasadas else list(tracks_by_pasada.keys())
        filters_to_process = selected_filters if selected_filters else list(available_filters.keys())
        
        total_combinations = 0
        for pasada in pasadas_to_process:
            if pasada in tracks_by_pasada:
                total_combinations += len(tracks_by_pasada[pasada]) * len(filters_to_process)
        
        print(f"\n[PLAN] Plan de procesamiento:")
        print(f"   Pasadas a procesar: {pasadas_to_process}")
        print(f"   Filtros a aplicar: {filters_to_process}")
        print(f"   Combinaciones totales: {total_combinations}")
        
        if args.dry_run:
            print(f"\n[INFO] DRY RUN - Mostrando que se procesaria:")
            for pasada in pasadas_to_process:
                if pasada not in tracks_by_pasada:
                    continue
                print(f"\n  Pasada {pasada}:")
                for track_file in tracks_by_pasada[pasada]:
                    track_name = os.path.basename(track_file)
                    print(f"    {track_name}")
                    for filter_name in filters_to_process:
                        output_path = create_output_path(filter_name, pasada, track_file)
                        exists = "OK" if os.path.exists(output_path) else "--"
                        print(f"      -> {filter_name:15} {exists} {output_path}")
            print(f"\nDRY RUN completado. Use sin --dry-run para procesar realmente.")
            return
        
        # Confirmar antes de procesar
        if total_combinations > 50:
            response = input(f"\n[WARN] Se van a procesar {total_combinations} combinaciones. Continuar? (y/N): ")
            if response.lower() != 'y':
                print("Operacion cancelada por el usuario.")
                return
        
        # Aplicar filtros
        apply_filters_to_tracks(
            tracks_by_pasada,
            available_filters,
            selected_filters=selected_filters,
            selected_pasadas=selected_pasadas,
            overwrite=args.overwrite,
            max_workers=args.max_workers
        )
        
        print(f"\n[DONE] Procesamiento completado!")
        print(f"   Los tracks filtrados estan en: results/filtered/<filtro>/<pasada>/")
        
    except KeyboardInterrupt:
        print(f"\n\n[WARN] Procesamiento interrumpido por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()



