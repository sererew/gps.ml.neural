Tengo una estructura de carpetas de esta forma: 
	data 
		raw 
			1 
			2 
			3 
			3a 
			... 
En cada capeta de tercer nivel (llamémosla "pasada") hay varios ficheros <nombre>.GPX y uno cuyo nombre es <n>_pattern_aligned_resampled.gpx. Llamemos a este ultimo el track patrón. 
Cada "pasada" corresponde a un recorrido hecho en determinado momento siendo grabado simultáneamente con varios dispositivos GPS con configuraciones diversas. 
Se llama "grabación" al track grabado por cada uno de los dispositivos al ejecutar la "pasada" siguiendo el "track patrón".
El track patrón contiene el recorrido realmente hecho (limpio, sin distorsiones ni ruidos). 
Las grabaciones corresponden a lo que realmente grabaron los aparatos y contienen ruido de diversa índole. 
El track patrón sirve de referencia para posiciones exactas en tiempos concretos.
Las grabaciones son imprecisas en posición, pero precisas en tiempo y hora (silmultáneas). 
Las grabaciones están muestreadas a 1 hz (tras un proceso de resamplizado)
El track patrón está también resamplizado a 1 hz. 


Para ello se necesita un script en python que lo haga.