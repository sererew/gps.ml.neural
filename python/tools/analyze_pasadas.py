import pandas as pd

# Cargar el manifest
manifest = pd.read_csv('data/input/manifest.csv')

print('=== ANÁLISIS DE PASADAS ===')
print(f'Total de ventanas: {len(manifest)}')

# Pasadas únicas
pasadas_unicas = sorted(manifest['pasada'].unique())
print(f'Pasadas únicas: {pasadas_unicas}')

# Contar ventanas por pasada
print('\n=== VENTANAS POR PASADA ===')
ventanas_por_pasada = manifest['pasada'].value_counts().sort_index()
print(ventanas_por_pasada)

# Contar grabaciones únicas por pasada
print('\n=== GRABACIONES ÚNICAS POR PASADA ===')
for pasada in pasadas_unicas:
    subset = manifest[manifest['pasada'] == pasada]
    grabaciones_unicas = subset['grabacion'].nunique()
    print(f'Pasada {pasada}: {grabaciones_unicas} grabaciones')

# Analizar patrón de nomenclatura
print('\n=== ANÁLISIS DEL PATRÓN <n><letra> ===')
pasadas_con_letra = []
pasadas_sin_letra = []

for pasada in pasadas_unicas:
    pasada_str = str(pasada)
    # Verificar si tiene sufijo de letra
    if any(char.isalpha() for char in pasada_str):
        pasadas_con_letra.append(pasada)
    else:
        pasadas_sin_letra.append(pasada)

print(f'Pasadas COMPLETAS (sin letra): {pasadas_sin_letra}')
print(f'Pasadas RECORTADAS (con letra): {pasadas_con_letra}')

# Mapear pasadas con letra a su base
print('\n=== RELACIÓN ENTRE PASADAS COMPLETAS Y RECORTADAS ===')
for pasada_letra in pasadas_con_letra:
    pasada_str = str(pasada_letra)
    # Extraer la parte numérica
    base_num = ''.join(filter(str.isdigit, pasada_str))
    letra = ''.join(filter(str.isalpha, pasada_str))
    
    if base_num:
        base_pasada = int(base_num)
        if base_pasada in pasadas_sin_letra:
            print(f'  {pasada_letra} ({letra}) -> derivada de pasada {base_pasada}')
        else:
            print(f'  {pasada_letra} ({letra}) -> ¡NO TIENE PASADA BASE {base_pasada}!')