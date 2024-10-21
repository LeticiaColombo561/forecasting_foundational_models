# Foundational Models 

Los modelos fundacionales son redes neuronales grandes, preentrenados, desiñados para comprender y generar varioas tipos de contenido. Estos modelos se enterenan con un conjunto de datos masivos, lo que permite capturar las complejidades y matices de los datos. Un grupo de Data Scientist del Stanford Institute for Human-Centered Artificial Intelligence (HAI) Center for Research on Foundation Models de Stanford, en la investigación On the Opportunities and Risks of Foundation Models, los definió como “cualquier modelo que se entrena con datos amplios (generalmente utilizando self-supervised a escala) que se puede adaptar (hacer fine-tuneing) a una amplia gama de tareas posteriores".  

Los modelos fundacionales se basan principalmente en self-supervised learning , de tal forma de que los grandes conjuntos de datos aprenden de sus propias etiquetas a partir de los datos de entrada. En el caso de los foundational models basados en texto, se incluyen en el aprendizaje gran cantidad de datos de diversas fuentes, en el caso de ser multimodales no solo aprenden del texto sino también de imágenes, audios y otros tipos de datos. 

Con estos modelos se puede hacer fine-tuning con nuestra data de manera de mejorar el ajuste. El conocimiento general y las características aprendidas durante el pre-entrenamiento se adaptan a tareas específicas que, en este caso son nuestros datos, que en general están etiquetados. 

El concepto de Foundational Models está basado en deep learning, es un subcampo de este, que aprovecha las redes neuronales. La adaptabilidad que tienen las redes permiten que los modelos fundacionales sean los suficientemente verátiles como para manejar entradas multimodales, la cual es una característica clave. 

En definitiva, los Foundational Models representan un tipo particular de modelo de Inteligencia Artificial, son modelos multimodales que combinan el entrenamiento previo mediante self-supervision, la arquitectura del deep-learning y la capacidad de poder hacer fine-tunning. 

Algunas características de los modelos fundacionales son el Transfer Learning y la arquitectura de Transformers. 

La idea detrás del Transfer Learning es aprovechar el conocimiento adquirido durante el entrenamiento inicial para mejorar el rendimiento de la nueva tarea. El hecho de que sean entrenados en un corpus muy diverso hace que se adapten más fácilmente a varias tareas y dominios. El fine-tuning entonces, permite que los modelos básicos se destaquen en una amplia gama de tareas. 

En cuanto a la arquitectura, los modelos básicos en general se basan en transformers, estos son un tipo de arquitectura de redes neuronales. Se basan en mecanismos de autoatención que permiten capturar dependencias y relaciones de largo alcance dentro de los datos, como por ejemplo las secuencias. Esta arquitectura fue la que sentó las bases de los Large Lenguaje Models. Otra particularidad es que no solo se aplican a lenguaje sino también a imágenes, gracias a la introducción de una variedad llamada Vision Transformer. 

##Arquitectura 

Veamos un poco más a fondo cómo funcionan los Transformers en general tienen un encoder y un decoder. En ambos podemos observar la presncia de un mecanismo de self-attention, una Fedd Forwar Network y capas de normalización. Veamos a continuación, que son cada una de ellas. 

  <IMG  src="https://fullstackdeeplearning.com/course/2022/lecture-7-foundation-models/media/image-3.png"  alt="alt_text"/>



### Self-attentnion 

Mecanismo que ayuda al modelo a determinar qué partes de una secuencia deben recibir más atención cuando se está procesando cada token. A diferencia de los modelos tradicionales que procesan secuencias en orden (por ejemplo, redes recurrentes), los transformers procesan todos los tokens de la secuencia en paralelo. Para hacer esto, necesitan un mecanismo para determinar qué partes de la secuencia son más relevantes para el token actual, y ahí es donde entra el self-attention. 

En un paso de self-attention, para cada token en la secuencia, el modelo realiza los siguientes pasos: 

1. Cálculo de tres vectores: Para cada token, el modelo aprende tres vectores: 

     - Query (Q): Representa la pregunta que se le hace a otros tokens (¿Qué información necesito?) 

     - Key (K): Representa la respuesta que cada token puede proporcionar (¿Qué información tengo?) 

     - Value (V): Es la información del token que será usada para el cálculo final. 

2) Cálculo de la atención: Para cada token, el transformer compara su Query con el Key de todos los demás tokens en la secuencia (incluido él mismo). Esto se hace calculando el producto escalar entre el Query del token y los Keys de los demás tokens, lo que produce una puntuación de atención para cada par de tokens. Esto indica qué tan "atentos" deben estar esos tokens entre sí. 

3) Aplicación de softmax: Las puntuaciones de atención se normalizan usando una función softmax para que todas sumen 1. Esto convierte las puntuaciones en probabilidades. 

4) Ponderación de los valores: Luego, cada valor (Value) de los tokens se pondera de acuerdo con estas puntuaciones de atención. De este modo, los tokens más relevantes (aquellos con mayores puntuaciones) tendrán más influencia en la representación final del token que se está procesando. 

3) Suma ponderada: Finalmente, se toma la suma ponderada de todos los valores de los tokens, y este resultado se convierte en la nueva representación del token actual. 

 
El mecanismo de multi-head self-attention se utiliza en varias capas. Esto significa que el modelo aprende múltiples representaciones de atención (o "cabezas") para cada token. Esto ayuda al modelo a enfocarse en diferentes partes del contexto simultáneamente, permitiéndole aprender patrones más complejos en la secuencia de entrada. 

 ### El positional encoding 

Es un componente fundamental en los transformers que complementa el mecanismo de self-attention. Su propósito es introducir información sobre la posición de los tokens en una secuencia, ya que el transformer, a diferencia de modelos secuenciales como las redes recurrentes (RNNs), no tiene una noción inherente de orden. 

Los transformers procesan toda la secuencia en paralelo, lo que significa que no asumen un orden natural entre los tokens. Sin embargo, en tareas como el procesamiento del lenguaje natural (NLP), el orden de las palabras es crucial para entender el significado de la oración. 

El positional encoding agrega una representación de las posiciones de los tokens a los embeddings de las palabras, de manera que cada palabra no solo está representada por su valor semántico, sino también por su posición en la secuencia. 

El método más común para calcular estos vectores de posición es usando funciones trigonométricas, específicamente senos y cosenos de frecuencias diferentes. El razonamiento detrás del uso de funciones sinusoidales es que proporcionan una forma continua y periódica de representar las posiciones, lo que permite al modelo aprender de secuencias de longitud variable y generalizar mejor a secuencias de diferentes tamaños. 

### Normalization Layer 

La Layer Normalization es una técnica de normalización que se aplica a las activaciones de una capa en la red neuronal. Su objetivo es estabilizar las salidas de la capa, mejorando la convergencia y facilitando el entrenamiento. En particular, la normalización en transformers ayuda a que el entrenamiento sea más eficiente y reduce el riesgo de problemas como la saturación de las activaciones o los gradientes. 

Los transformers son arquitecturas profundas que dependen de bloques repetidos de elf-attention y redes feed-forward. Esto puede hacer que las activaciones en las capas posteriores se vuelvan muy grandes o muy pequeñas, lo que podría causar problemas en el entrenamiento. La Layer Normalization mitiga esto normalizando las activaciones. 

En la arquitectura de transformers, las capas de normalización se colocan generalmente de dos maneras: 

Antes del mecanismo de self-attention: Se normalizan las entradas antes de aplicar la autoatención, lo que asegura que las entradas a la capa de atención estén bien equilibradas y estabilizadas. 

Después del bloque feed-forward: También se usa la normalización después del paso feed-forward para estabilizar la salida y antes de agregar la conexión residual (skip connection). 

### Encoder y Decoder 

El **encoder** es responsable de procesar la secuencia de entrada y generar una representación codificada que captura la información y relaciones contextuales entre los tokens (o palabras) de la entrada. El encoder es particularmente útil para tareas donde se necesita un entendimiento profundo del contenido. 

El flujo del encoder se puede ver de la siguiente forma: 

1. Cada token de la secuencia de entrada se embebe en un vector. 
2. A estos embeddings se les suman los positional encodings (codificaciones posicionales). 
3. Se aplican múltiples capas de atención propia y redes feed-forward para obtener una representación rica del texto de entrada. 
4. El decoder es responsable de generar la secuencia de salida basándose tanto en la representación codificada generada por el encoder como en las palabras generadas anteriormente.  

La particularidad del **decoder** es que además de tener un mecanismo Self-Attention, que tiene la particularidad de que solo puede prestar atención a tokens previos pero no a los futuros, gracias al positonal encoder, para asegurarse de que no “hace trampa” mirando las palabras futuras, también tiene un self attention sobre la salida de Ecoder. 

Esta segunda capa de atención, permite que el decoder "preste atención" a la representación codificada de la secuencia de entrada. Aquí, el decoder toma la secuencia codificada del encoder y selecciona las partes relevantes para generar la palabra o token actual en la salida. 

1. El flujo del decoder se puede ver de la siguiente forma: 
2. Masked Multi-Head Attention: Procesa los tokens generados hasta el momento, permitiendo que cada token "preste atención" a tokens previos. 
3. Multi-Head Attention con conexión al encoder: Permite que el decoder use la representación de la secuencia de entrada generada por el encoder. 
4. Feed-Forward Network: Refina las representaciones de los tokens. 

Por ultimo se agregan dos capas:

1. Capa Linear: Transforma las representaciones en puntuaciones para cada palabra del vocabulario. 
2. Softmax: Convierte esas puntuaciones en probabilidades, de las cuales se selecciona la palabra/token más probable. 

 

## Recursos 

https://aws.amazon.com/what-is/foundation-models/ 

https://arxiv.org/pdf/2108.07258 

https://toloka.ai/blog/the-power-of-foundation-models/ 

https://fullstackdeeplearning.com/course/2022/lecture-7-foundation-models/ 

 

