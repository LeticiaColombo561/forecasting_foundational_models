# TimeGPT 

Históricamente, los métodos estadísticos como ARIMA, SARIMA y Multiple-Seasonal-Trend, etc. y los modelos de machine learning como  XGBoost y LightGBM han sido usados con éxito para hacer predicciones de series temporales. Sin embargo, los modelos de deep learning ofrecen ventajas en escalabilidad, flexibilidad y capacidad para capturar dependencias complejas de los datos, lo que los hace atractivos para tareas de pronóstico con grandes volúmenes de datos. 

A pesar de ello, algunos profesionales cuestionan su superioridad, presentando evidencia de que modelos más simples pueden superar a los sofisticados en términos de precisión y eficiencia. Las barreras incluyen la falta de conjuntos de datos de prueba adecuados y configuraciones de evaluación estandarizadas para series temporales.   

TimeGPT surge como un modelo innovador que, con menor complejidad, supera a las alternativas existentes, resaltando la importancia de investigar más los modelos fundacionales que puedan mejorar el pronóstico de series temporales.  

En este contexto, TimeGPT se presenta como una innovación significativa. Es el primer modelo de base que logra superar de manera consistente a las alternativas con una complejidad mínima. TimeGPT se posiciona como el comienzo de un nuevo capítulo en el análisis de series temporales al demostrar que, con conjuntos de datos más grandes y diversos, los modelos de deep learning  pueden ofrecer mejoras tangibles. Este avance sugiere que una mayor investigación en modelos fundacionales y conjuntos de datos adecuados podría potenciar el campo de las prediciones de series temporales, mejorando tanto la precisión como la eficiencia en futuras aplicaciones. 

Azul Garza and Max Mergenthaler-Canseco de Nixtla, mostraron la arquitectura, entrenamiento y evaluación del modelo Time-GPT en el paper "TimeGPT-1". Allí comparan la performance de los distintos modelos, destacando que es user-friendly, low-code y se puede realizar predicciones desde cero, con series que no han sido vistas antes. 

Como todos estos foundational models, los datos se leen como sentencias, “tokens” y predicen que viene luego. Estas predicciones están basadas en patrones que el modelo identifica en la data del pasado y lo extrapola al presente. 

Comparando con modelos estadísticos, machine leararning y deep learning models, TimeGPt tiene mejor performances, eficiencia y simplicidad en inferencia Zero-Shoot según lo que se muestra en el paper. 


## La arquitectura del modelo 
![image.png](/.attachments/image-70144252-620e-499b-ac9c-4b70881e56fc.png)
 
### Entradas 

1. Target Variable (Variable objetivo): 

   La serie temporal principal que se quiere predecir. Esto es el conjunto de datos históricos sobre el cual se entrena el modelo para realizar predicciones futuras. 

2. Events (Eventos): 

   Representan posibles factores externos o eventos importantes (como festivos, promociones, o cambios abruptos) que pueden influir en la variable objetivo. 

3. Additional Variables (Variables adicionales): 

   Variables exógenas que se usan como entradas adicionales para el modelo. Estas podrían incluir otras series temporales relevantes o variables contextuales (precio, clima, etc.) que afecten el comportamiento de la variable objetivo. 

### Procesamiento 

1. Bloque de Encoding: 

   a. Input Embedding: transforma los datos de entrada en una representación más adecuada para que el modelo pueda procesarlos. 

   b. Positional Encoding: en el primer bloque, la codificación posicional se aplica a las entradas de la serie temporal (a los embeddings de la variable objetivo y las variables adicionales). 
  
   c. Las entradas originales, como la serie temporal objetivo y las variables exógenas, se combinan con el positional encoding para que el modelo entienda la secuencia temporal. 

   d. Este bloque usa una Feed Forward Network para procesar los patrones locales de las series temporales. Aquí es donde se detectan ciclos, tendencias estacionales, y otros patrones en las secuencias de datos. Se extraer características más abstractas y no lineales que faciliten el pronóstico. 

2. Bloque de Decoding:  

   a. Positional Encoding del Output Embedding: este segundo Positional Encoding se aplica a los embeddings de salida, es decir, al espacio vectorial que representa las secuencias de salida. Este proceso permite que el modelo entienda dónde se encuentra cada valor previsto dentro de la secuencia de tiempo que se está prediciendo.  

   b. Masked Multi-Head Attention: El término Masked hace referencia a que no puede mirar hacia adelante en el tiempo. En este caso en particullar, cuando estamos en un momento en el tiempo t, no permite que se utilice información futura. 

   d. El término Multi-Head Attention permite que el modelo se "enfoque" en diferentes partes de la secuencia de entrada para cada predicción, capturando tanto relaciones a corto plazo como a largo plazo. Pero al ser "enmascarada", el modelo está limitado a mirar solo en el pasado y no en el futuro, garantizando que no se use información futura en las predicciones. 

   e. El Multi-Head Attention que combina información del encoder junto con la FFN  ayuda al modelo a aprender patrones de las entradas (como eventos, variables adicionales y la serie objetivo) mientras mantiene información sobre el orden de los datos. 

   f. Por último, las capas Linear y Softmax, transforman las representaciones en puntuaciones con sus probabilidades para luego ser seleccionadas. 

 

### Salidas 

Forecast: El modelo finalmente genera predicciones o pronósticos sobre la serie temporal. La parte final muestra la predicción futura que el modelo realiza. 

 

## Entrenamiento de TimeGPT 

TimeGPT fue entrenado utilizando la colección más grande de series temporales disponibles públicamente, abarcando más de 100 mil millones de puntos de datos provenientes de diversos dominios como finanzas, salud, clima, energía, entre otros. Es un conjunto de datos es extremadamente diverso en términos de patrones temporales, estacionalidades, tendencias y niveles de ruido, lo que permite al modelo aprender de una amplia gama de escenarios. Gracias a esta diversidad, TimeGPT puede generalizar y hacer pronósticos precisos sobre series temporales no vistas, según lo que se comentan en el paper, eliminando la necesidad de entrenar modelos individuales para cada situación. 

El proceso de entrenamiento de TimeGPT se realizó en un cluster de GPUs NVIDIA A10G, con una exploración exhaustiva de hiperparámetros, como la tasa de aprendizaje y el tamaño de lote. Se descubrió que un tamaño de lote mayor y una tasa de aprendizaje más baja ofrecían los mejores resultados. El modelo fue implementado en PyTorch y entrenado con el optimizador Adam, aplicando una estrategia de reducción de la tasa de aprendizaje al 12% de su valor inicial, lo que permitió un ajuste más fino durante el proceso de aprendizaje. 

Además de realizar predicciones puntuales, TimeGPT también incorpora predicción probabilística para estimar la incertidumbre en sus pronósticos, lo que es crucial para la toma de decisiones informada y la evaluación de riesgos. Utiliza un enfoque de predicción que genera intervalos de predicción con un nivel preestablecido de precisión de cobertura. A diferencia de los métodos tradicionales, este enfoque no requiere suposiciones estrictas sobre la distribución de los datos, lo que lo hace flexible y aplicable a distintos dominios. 

Durante la inferencia de nuevas series temporales, TimeGPT realiza pronósticos continuos basados en los datos más recientes, ajustando los errores del modelo de manera dinámica. Esto mejora la precisión de las predicciones al permitir que el modelo se adapte constantemente a las últimas tendencias y cambios en los datos, lo que lo convierte en una herramienta eficaz para el pronóstico de series temporales en un amplio espectro de aplicaciones. 

## Resultados 

Tradicionalmente los modelos se evalúan dividiendo las series temporales en conjuntos de entrenamiento y prueba, pero este enfoque no es suficiente para un modelo fundacional como TimeGPT, cuya fortaleza es predecir series completamente nuevas. TimeGPT fue probado en más de 300 mil series temporales de diversos dominios, como finanzas, tráfico web, clima, IoT, y electricidad, sin haber visto estos datos durante su entrenamiento. 

La evaluación se realizó en la última ventana de pronóstico de cada serie, ajustando el horizonte de predicción según la frecuencia de los datos (por ejemplo, 12 meses para datos mensuales o 24 horas para datos horarios). El modelo no fue reentrenado (zero-shot) y utilizó solo datos históricos anteriores como entrada. El rendimiento de TimeGPT se comparó con varios modelos de referencia y estadísticos, excluyendo algunos como Prophet y ARIMA debido a sus altos requisitos computacionales. Las métricas de evaluación seleccionadas fueron rMAE y rRMSE, que se normalizaron en relación con el modelo Seasonal Naive, lo que facilitó la comparación entre diferentes frecuencias de datos. 

En la prueba de inferencia de Zero-shot, TimeGPT mostró resultados impresionantes, superando a muchos modelos estadísticos y de aprendizaje profundo, y posicionándose entre los tres mejores en todas las frecuencias de datos evaluadas. Esto demuestra su capacidad para ofrecer pronósticos precisos sin necesidad de ajustes adicionales. Además, se destacan factores como el costo computacional y la simplicidad en la implementación, en estos aspectos TimeGPT también mostró un rendimiento eficiente. 

 ![image.png](/.attachments/image-c9b53b21-9419-4fb5-b59d-5eb51d47a8bb.png)


## Funcionalidades de TimeGPT

- Anomaly Detection: El modelo tiene la capacidad de detectar anomalías en los datos, identifica puntos de datos fuera del comportamiento normal, ayudando a detectar actividades fraudulentas, violaciones de seguridad o valores atípicos significativos. La detección de anomalías implica hacer predicciones y generar un intervalo de confianza del 99%. Si un punto observado cae fuera de ese intervalo, es una anomalía.

- Long Horizon Forecasting: TimeGPT permite realizar predicciones a largo plazo, utilizando el modelo "timegpt-1-long-horizon". El pronóstico de largo horizonte se refiere a predicciones en el futuro, que generalmente superan los dos períodos estacionales. Por ejemplo, para los datos diarios, un pronóstico que abarca más de dos semanas cae en la categoría de horizonte largo.

- Azure Integration: TimeGPT ofrece una solución que está integrada con Azure accediendo a través de AzureAI. Se puede implementar como un servicio con pago por uso a través de Azure AI, proporcionando una forma de consumir el modelo como una API sin alojarlos en su suscripción, al tiempo que mantiene la seguridad empresarial.

- Agregado de Variables Exógenas: TimeGPT permite la incorporación de variables exógenas, como días festivos, variables categóricas y otros factores externos, para enriquecer el modelo y mejorar la precisión de las predicciones. También incluye el uso de valores SHAP (SHapley Additive exPlanations) para analizar la importancia de estas variables en el modelo. 

- Validación mediante Cross-Validation: Para asegurar la robustez de las predicciones, TimeGPT admite la validación cruzada (cross-validation), lo que permite evaluar el rendimiento del modelo en diferentes subconjuntos de datos y asegurar que las predicciones sean consistentes.

- Intervalos de confianza: TimeGPT incluye la posibilidad de estimar los resultados con intervalos de confianza. Estos permiten capturar la incertidumbre en las predicciones, proporcionando un rango probable para los valores futuros en lugar de solo una estimación puntual, lo que hace que las predicciones sean más robustas.

- Métricas para Evaluar Resultados: TimeGPT incluye diversas métricas para evaluar los resultados de las predicciones, lo que facilita la comprensión de su rendimiento y la identificación de áreas de mejora.

- Finetuning con Funciones de Pérdida Específicas: TimeGPT ofrece la opción de realizar fine-tuning para adaptar el modelo a un nuevo conjunto de datos. Este proceso incluye varios parámetros configurables:

   - finetune_steps: Define el número de pasos para el fine-tuning, permitiendo personalizar la extensión del entrenamiento en nuevos datos.
   - finetune_loss: Permite elegir la función de pérdida a utilizar durante el el fine-tuning, con opciones que incluyen errores absolutos (MAE), errores cuadráticos (MSE), y errores porcentuales (MAPE), entre otros.


## Conclusiones

TimeGPT ha demostrado ser una herramienta altamente eficaz y versátil para el pronóstico de series temporales, no solo por su capacidad para ofrecer predicciones precisas en una amplia gama de dominios y frecuencias, sino también por su eficiencia computacional y facilidad de implementación. Además cuenta con la posibilidad de las zero-shot, que permite realizar pronósticos sin necesidad de re-entrenar el modelo para nuevas series temporales. A su vez se puede hacer fácilmente fine-tuning y se puede implementar cross-validation para tener mayor robustez en los resultados. Por último se puede destacar la integración que tiene con Azure proporcionando mayor seguridad. 
 



 

## Recursos 

- https://arxiv.org/pdf/2310.03589 

- https://www.analyticsvidhya.com/blog/2024/02/timegpt-revolutionizing-time-series-forecasting/ 

- https://nixtlaverse.nixtla.io/nixtla/docs/use-cases/intermittent_demand.html#forecasting-with-exogenous-variables-using-timegpt 

- https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/crossvalidation.html 

- https://www.nixtla.io/news/timegen1-on-azure


