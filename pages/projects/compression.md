# Correlation Compressibility

Scientific research increasingly uses error-bounded lossy compressors to achieve greater compression ratios in relation to lossless compressors. This improved performance allows applications to run with larger and more frequently produced datasets due to faster I/O times and smaller I/O volumes. The theoretical limit on compressibilty of data using lossless compression is given by the entropy. The entropy quantifies the information content present in a symbol from a source sequence based upon its probability of occurrence. Thus, for a given sequence of symbols the entropy enables computing the minimum number of bits, on average, needed to represent the data. 

### Motivation 

For over 70 years, this concept has guided the development and evaluation of lossless compression algorithms. However, for lossy compression algorithms, there is currently no known bound for the maximum degree of compression that can be achieved for some specified point-wise error bound regardless of the compressor at stake. 
Establishing this compressor-free bound will allow researchers to anticipate compression performance alleviating manual assessments. 
This bound can be used to evaluate with respect to this compressor-free roofline and adapt existing compressors to correlation structures of the data ensuring they get the best compression ratio possible. 
Eventually, establishing the limit for lossy compression allows for the maximum efficiency for storing large scientific datasets. 

### Proof of concept

SZ, ZFP, and MGARD are some of the leading lossy compressors. Each compressor uses different features of local and global correlation to generate a compressed representation of data. The first paper[^1] focused on statistical variogram study of correlations present within differing lossy compressors. The variogram is described below.

$$
\displaystyle \gamma(h) = \frac{1}{2N(h)}\sum_{|x_i - x_j|=h}^{N(h)}\left ( z(x_i)-z(x_j) \right)^2
$$

where $z$ is the studied field of interest (e.g. $\textit{velocityx}$ from the Miranda dataset), $x_i$ and $x_j$ are grid-points coordinates or indexes, $N(h)$ is the number of points at distance $h$ from each other. 

![ ](/assets/images/projects/compression/variogram.png)

In this figure, compression ratios are compared against standard deviation of the local variogram range for single range correlation Gaussian fields (left) and for multi-
range correlation Gaussian fields (right). This showcases an initial step towards the use of
decomposition techniques to analysis fields with multiscale patterns. 

However, results for the single-range correlation Gaussian fields show a weaker sensitivity of the compression ratios to this local statistic. This might suggest the need to use several statistics. 

![ ](/assets/images/projects/compression/svd-trunc.png)

In this figure, compression ratios are compared against standard deviation of the truncation level of local SVD for single scale Gaussian fields (left) and for multiscale Gaussian
datasets (right). Local SVDs are performed on the fields and summarized via the standard deviation of locally required numbers of singular modes to capture 99% of the variance of each local window.

Larger values of required singular modes are associated with less compressible fields so we expect mostly decreasing relationship of compression ratios to this local statistic. 


### Modeling [![GitHub](https://github.githubassets.com/favicons/favicon.svg)](https://github.com/FTHPC/Correlation_Compressibility)
The next step is to explore more complex dependent variables as predictors and create a model of the compression ratio based on correlation metrics and error bound. 
This was the innitative of the second paper[^2]. We found the following regression model to perform quite well. 

$$
\begin{eqnarray}
  \nonumber  \log(\mbox{CR}) &=& a + b\times\log(\mbox{q-ent}) +  c\times\log\left(\frac{\mbox{SVD-trunc}}{\sigma}\right) \\  
   \nonumber  && +  d\times\log(\mbox{q-ent})\times\log\left(\frac{\mbox{SVD-trunc}}{\sigma}\right)+\epsilon \\
    && 
\end{eqnarray}  
$$


#### SVD-trunc
One of the main focuses of the above regression model is the SVD-trunc statistic. Introduced in the discussion about the first paper, we condier the percentage of singular values needed to recover 99% of the total variance of matrix X. We denote this as SVD-trunc. This statistic is meant to showcase the spatial correlation within the field. High truncation levels indicate that a high number of singular modes is required to capture most of the variabiliity. 

One issue here is we are working with 3D datasets. We will focus on the high-order SVD (HOSVD). HOSVD is calculated by unfolding dimension-i to arrange the dimension i-columns into a 2D-matrix where a traditional SVD is used. This will result in i different singular values diagonals. Our method operates by sorting and selecting singular values from each unfolded matrix such that the total contribution of the squared signular values exceeds 90% per dim. 

#### q-ent
Entropy is viewed as an upper bound on the compression ratio for lossless compressors. It provides a good estimate of occurrence of symbols within a dataset. Entropy does not account for loss of information. This is where we quantize the entropy to produce some notion of loss. 

![ ](/assets/images/projects/compression/prediction.png)

The graph showcases out-of-sample prediction accuracy for compression ratio for SZ2 and ZFP on a QMCpack dataset. It performs well; however, there are some future steps identified throughtout the findings in this paper. Future work should focus further on the generalizability of the method to reduce dependence on samples, types of bounds, and compressors principles. This is explored in the following paper[^3], though I will not discuss it in detail here.

[^1]: [Paper](/assets/documents/papers/2021/Exploring_Lossy_Compressibility_through_Statistical_Correlations_of_Scientific_Datasets.pdf) D. Krasowska, J. Bessac, R. Underwood, J. C. Calhoun, S. Di, and F. Cappello. "Exploring Lossy Compressibility through Statistical Correlations of Scientific Datasets," 2021 7th International Workshop on Data Analysis and Reduction for Big Scientific Data (DRBSD-7), St. Louis, MO, USA, 2021, pp. 47-53, doi:10.1109/DRBSD754563.2021.00011
[^2]: [Paper](/assets/documents/papers/2023/Black-Box_Statistical_Prediction_of_Lossy_Compression_Ratios_for_Scientific_Data.pdf) R. Underwood, J. Bessac, D. Krasowska, J. C. Calhoun, S. Di, and F. Cappello. "Black-box statistical prediction of lossy compression ratios for scientific data," The International Journal of High Performance Computing Applications (IJHPCA), 2023, pp. 412-433, doi:10.1177/10943420231179417  
[^3]: [Paper](/assets/documents/papers/2023/A_Lightweight_Effective_Compressibility_Estimation_Method_for_Error-bounded_Lossy_Compression.pdf) A. Ganguli, R. Underwood, J. Bessac, D. Krasowska, J. C. Calhoun, S. Di, and F. Cappello. "A Lightweight, Effective Compressibility Estimation Method for Error-bounded Lossy Compression," IEEE International Conference on Cluster Computing (CLUSTER), Santa Fe, NM, 2023, pp. 247-258, doi:10.1109/CLUSTER52292.2023.00028 