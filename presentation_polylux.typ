#import "@preview/polylux:0.4.0": *

#set page(paper: "presentation-16-9")
#set text(font: "New Computer Modern", size: 20pt)
#show math.equation: set text(font: "New Computer Modern Math")
#set strong(delta: 100)

// Enable speaker notes
#enable-handout-mode(false)

// Simple theme colors
#let primary-color = rgb("#1a5490")
#let accent-color = rgb("#333333")
#let text-color = rgb("#333333")

// Title slide
#slide[
  #align(center + horizon)[
    #v(-2cm)
    #image("university_of_cambridge.png", width: 35%)
    
    #v(-2cm)
    #text(size: 32pt, weight: "bold")[
      MPhil in Data Intensive Science
    ]
    
    
    #text(size: 26pt)[A7 Image Analysis Coursework]
    
    #v(1.5cm)
    
    #text(size: 22pt)[*Bocheng Xiao* (bx242)]
    
    #text(size: 18pt, fill: gray)[July 2025]
  ]
]

// Table of Contents
#slide[
  #align(center)[
    #text(size: 32pt, weight: "bold")[Overview]
  ]
  
  #v(1cm)
  
  #grid(
    columns: (1fr, 1fr),
    gutter: 3em,
    [
      #text(weight: "bold", fill: accent-color)[Module 1: Classical Image Processing]
      - Color Classification
      - Background Removal  
      - Collection Display
      - Odd One Out Detection
    ],
    [
      #text(weight: "bold", fill: accent-color)[Module 2: Image Restoration]
      - PnP-ADMM Algorithm
      - Deblurring & Inpainting
      - Overfitting Discovery
      
      
      #text(weight: "bold", fill: accent-color)[Module 3: Quality Assessment]
      - IQA Metric Analysis
      - ML System Debugging
    ]
  )
]

// Module 1 Title
#slide[
  #align(center + horizon)[
    #text(size: 48pt, weight: "bold")[Module 1]
    
    
    #text(size: 32pt)[Classical Image Processing]
    
    #text(size: 20pt, fill: gray)[
      Butterfly Classification & Analysis
    ]
  ]
]

// Color Classification
#slide[
  #text(size: 28pt, weight: "bold")[Color Classification]
  
  *HUE-Based Approach:*
  #figure(
        image("images/task11.png", width: 100%),
      )
]

// Classification Results
#slide[
  #text(size: 28pt, weight: "bold")[Color Classification]
  
  #grid(
    columns: (1.4fr, 1fr),
    gutter: 2em,
    [
      #figure(
        image("images/task112.png", width: 100%),
      )
    ],
    [
      *Methodology*:
      1. dominant HUE classification
      2. quantile-based grouping
      3. actually first remove the backgrounds then classify
      #align(center)[
        #table(
          columns: 2,
          inset: 10pt,
          stroke: 0.5pt + primary-color,
          fill: (x, y) => if x == 0 { primary-color.lighten(90%) },
          [*Red-Orange*], [9.0° - 16.0°],
          [*Yellow*], [23.0° - 29.0°],
          [*Blue*], [96.0° - 114.0°],
        )
        #text(size: 20pt, weight: "bold", fill: accent-color)[100% Success Rate (12/12)]
      ]
    ]
  )
]

// Background Removal Results
#slide[
  #align(center)[
    #text(size: 28pt, weight: "bold")[Background Removal Results]
    
    #columns(3, gutter: 1em)[
      #figure(
        image("images/task12_1.png", width: 92%),
      )
      
      #colbreak()
      
      #figure(
        image("images/task12_2.png", width: 92%),
      )
      
      #colbreak()
      
      #figure(
        image("images/task12_3.png", width: 92%),
      )
    ]
    #text(size: 22pt, weight: "bold", fill: accent-color)[All butterfly image backgrounds are well removed.]
    
  ]
  
]

// Background Removal
#slide[
  #text(size: 28pt, weight: "bold")[Background Removal]
  
  #columns(2)[
    *Multi-Algorithm Approach:*
    
    #text(fill: accent-color, weight: "bold")[Primary: GrabCut Auto]
    - Initial rect: 10% margin from edges
    - GMM models foreground/background
    - 5 iterations for convergence
    - Success rate: 85% of images
    
    #text(fill: accent-color, weight: "bold")[Fallback Chain]
    
    *1. GrabCut Center Bias*
    - Assumes butterfly in center
    - Marks edges as definite background
    
    *2. Color Segmentation*
    
    *3. Watershed*

    
    #text(fill: accent-color, weight: "bold")[Performance Metrics:]
    
    #table(
      columns: 2,
      inset: 10pt,
      stroke: 0.5pt + primary-color,
      fill: (x, y) => if x == 0 { primary-color.lighten(90%) },
      [*Success Rate*], [100%],
      [*Avg Quality*], [0.913 ± 0.041],
      [*Edge Fidelity*], [High],
    )
  ]
]

// Quality Metrics
#slide[
  #text(size: 28pt, weight: "bold")[Quality Metrics System]
  
  #rect(
    width: 100%,
    fill: primary-color.lighten(85%),
    stroke: 1pt + primary-color,
    radius: 8pt,
    inset: 15pt
  )[
    #text(fill: accent-color, weight: "bold")[Foreground Ratio (Butterfly should fill around 40% of image)]

    *Measure:* Count foreground pixels / total pixels; 
    #v(-0.5cm)
    *Score:* Perfect at 40% |
    Zero at \<5% or \>95%
    
    *Why:* Too small = lost details |
    Too large = cropped wings
    
    
    
    
  ]
  #v(-0.5cm)
  #rect(
    width: 100%,
    fill: primary-color.lighten(85%),
    stroke: 1pt + primary-color,
    radius: 8pt,
    inset: 15pt
  )[
    #text(fill: accent-color, weight: "bold")[Connected Components (Single butterfly object)]
    
    *Why:* Multiple parts = broken segmentation
    
    *Measure:* cv2.findContours() to count separate regions
    #v(-0.5cm)
    *Score:* 1 component = 1.0 |
    10+ components = 0.5
  ]
  
  #rect(
    width: 100%,
    fill: primary-color.lighten(85%),
    stroke: 1pt + primary-color,
    radius: 8pt,
    inset: 15pt
  )[
    #text(fill: accent-color, weight: "bold")[Main Object Size (Largest piece should $≥33%$ of image)]
    
    *Why:* Ensures butterfly is the main object, not noise
    
    *Measure:* Largest contour area / image area
    
    *Score:* Max at 33%+ |
    Penalizes scattered pixels
  ]
  #rect(
    width: 100%,
    fill: accent-color.lighten(90%),
    stroke: 1pt + accent-color,
    radius: 5pt,
    inset: 6pt
  )[
    *Final Score = (Ratio Score + Component Score + Largest Score) / 3*

    *Threshold:* Score > 0.7 → Accept segmentation | Score ≤ 0.7 → Try next method
  ]
]



// Collection Display
#slide[
  #text(size: 28pt, weight: "bold")[Collection Display System]
  
  #columns(2, gutter: 2em)[
      
    #figure(
      image("images/task131.png", width: 100%),
    )

    #grid(
    columns: 2,
    gutter: 1em,
    figure(
      image("images/task132.png", width: 100%),
    ),
    figure(
      image("images/task133.png", width: 100%),
    )
    )
    
    
    #colbreak()
    
    1. Optimal grid layout:
       ```python
       images_per_row = ceil(sqrt(n))
       ```
    2. Smart resizing with bounding box:
       - Find and crop to bounding box
       - Scale to fit cell
    
    3. Alpha blending algorithm:
       ```python
       result = alpha * butterfly + 
                (1-alpha) * background
       ```
    4. Multiple Backgrounds:
       - Solid / Gradient / Textured
  ]
]

#slide[
  #text(size: 28pt, weight: "bold")[Odd One Out Detection]
  #figure(
      image("images/task14.png", width: 51%),
      caption: "100% accuracy in odd detection"
    )
]
// Odd One Out
#slide[
  #text(size: 28pt, weight: "bold")[Odd One Out Detection]
  
  #columns(2, gutter: 2em)[
    *8-Feature Color Analysis:*
    
    1. *Black detection* (5× weight):
       - 5 methods: HSV, RGB, dark edges
       - Amplifies ANY black presence
    
    2. *Dual-tone score* (4.5× weight):
       - Yellow+black vs pure yellow
       - Key for Group_2_Yellow
    #colbreak()
    3. *Multi-color complexity*:
       - Counts significant color bins
       - Special yellow+black bonus
    
    4. *Pattern contrast*:
       - Sobel gradient analysis
       - Detects sharp transitions
    
    *Outlier Detection:*
    - Weighted distance calculation
    - Max avg distance = odd one
  ]
]

// Module 2 Title
#slide[
  #align(center + horizon)[
    #text(size: 48pt, weight: "bold")[Module 2]
    
    
    #text(size: 32pt)[Plug-and-Play ADMM]
    
    #text(size: 20pt, fill: gray)[
      Image Restoration via Hybrid Optimization
    ]
  ]
]

// PnP-ADMM Theory
#slide[
  #text(size: 28pt, weight: "bold")[PnP-ADMM Framework]
  
  #columns(2, gutter: 2em)[
    *Optimization Problem:*
    #rect(fill: primary-color.lighten(95%), inset: 10pt)[
      $ min_x 1/2 ||A x - y||_2^2 + g(x) $
    ]
    
    where:
    - $A$: Forward operator (blur/mask)
    - $y$: Observed data
    - $g(x)$: Regularization term
    
    *Key Innovation:* 
    Replace $"prox"_g(·)$ with pre-trained U-Net denoiser $D(·)$
    
    #colbreak()
    
    *ADMM Iterations:*
    
    #text(fill: accent-color, weight: "bold")[1. x-update (Data fidelity)]
    $x^(k+1) = (A^T A + eta I)^(-1)(A^T y + eta(v^k - u^k))$
    
    #text(fill: accent-color, weight: "bold")[2. v-update (Denoising)]
    $v^(k+1) = D(x^(k+1) + u^k)$
    
    #text(fill: accent-color, weight: "bold")[3. u-update (Dual variable)]
    $u^(k+1) = u^k + (x^(k+1) - v^(k+1))$
    
    Parameter: $eta = 10^(-4)$
  ]
]

// Task 2.1.1: Motion Blur
#slide[
  #text(size: 28pt, weight: "bold")[Task 2.1.1: Motion Blur Deblurring]
  
  #grid(
    columns: (1.4fr, 1fr),
    gutter: 2em,
    [
      #figure(
        image("images/task21.png", width: 100%),
      )
    ],
    [
      *Motion Blur Operator:*
      ```python
      kernel = np.zeros((p, p))
      kernel[p//2, :] = 1/p
      ```
      
      #table(
        columns: 3,
        inset: 8pt,
        stroke: 0.5pt + primary-color,
        fill: (x, y) => if y == 0 { primary-color.lighten(90%) },
        [*Kernel*], [*MSE*], [*PSNR*],
        [p = 7], [5.2e-5], [42.81 dB],
        [p = 13], [1.19e-4], [39.26 dB],
        [p = 17], [1.67e-4], [37.77 dB],
      )
      #rect(
        fill: accent-color.lighten(90%),
        stroke: 1pt + accent-color,
        radius: 5pt,
        inset: 7pt
      )[
      *Finding:* Performance degrades gracefully with larger kernels
      ]
    ]
  )
]

// Task 2.1.2: Effect of Noise
#slide[
  #text(size: 28pt, weight: "bold")[Task 2.1.2: Effect of Gaussian Noise]
  
  #grid(
    columns: (1.4fr, 1fr),
    gutter: 2em,
    [
      #figure(
        image("images/task212.png", width: 100%),
      )

      #rect(
        fill: accent-color.lighten(90%),
        stroke: 1pt + accent-color,
        radius: 5pt,
        inset: 7pt
      )[
        *Finding:* Noise degrades performance as expected - the algorithm behaves consistently with theory
      ]
    ],
    [
      *Experimental Setup:*
      - Motion blur: p = 13
      - Add Gaussian noise: σ = 0.01
      - Compare clean vs noisy blur
      
      #table(
        columns: 2,
        inset: 8pt,
        stroke: 0.5pt + primary-color,
        fill: (x, y) => if x == 0 { primary-color.lighten(90%) },
        [*Condition*], [*PSNR (dB)*],
        [Clean blur], [39.26],
        [Blur + noise], [25.88],
      )
      
      *Performance Impact:*
      - 13.38 dB degradation
      - MSE: 1.19e-4 → 2.58e-3
      
    ]
  )
]

// Task 2.2.1: Inpainting
#slide[
  #text(size: 28pt, weight: "bold")[Task 2.2.1: Image Inpainting]
  
  #grid(
    columns: (1fr, 1.5fr),
    gutter: 2em,
    [ 
      #table(
        columns: 3,
        inset: 10pt,
        stroke: 0.5pt + primary-color,
        fill: (x, y) => if y == 0 { primary-color.lighten(90%) },
        align: (center, auto, auto),
        [*Missing*], [*MSE*], [*PSNR*],
        [40%], [4.9e-5], [43.07 dB],
        [60%], [9.6e-5], [40.18 dB],
        [80%], [1.87e-4], [37.28 dB],
      )
      #rect(
        fill: accent-color.lighten(90%),
        stroke: 1pt + accent-color,
        radius: 5pt,
        inset: 7pt
      )[
      *Finding:* 
      1. Performance degrades with larger missings
      2. Robust recovery even with 80% missing pixels
      ]
    ],
    [
      #figure(
        image("images/task221.png", width: 100%),
        caption: "From sparse samples to full reconstruction"
      )
    ]
  )
]

// Task 2.2.2: PnP-RED
#slide[
  #text(size: 28pt, weight: "bold")[Task 2.2.2: PnP-RED Algorithm]
  
    *RED Framework:*
    
    Regularizer from denoiser:
    $rho(x) = 1/2 x^T(x - D(x))$
    
    Gradient descent for:
    $J(x) = 1/2 ||y - A x||_2^2 + lambda rho(x)$
    
    Update rule:
    $x^(k+1) = x^k - eta nabla J(x^k)$
    
    where:
    $nabla J(x) = A^T(A x - y) + lambda(x - D(x))$
    
    *Implementation (60% missing):*
    - Step size: η = 1.0
    - Regularization: λ = 0.1
    
]

#slide[
    #text(size: 28pt, weight: "bold")[Task 2.2.2: PnP-RED Algorithm]
    #grid(
      columns: (2fr, 1fr),
      gutter: 1.4em,
      [
        #figure(
          image("images/task222.png", width: 100%),
          caption: "PnP-ADMM vs PnP-RED comparison"
        )
      ],
      [
        #table(
          columns: (auto, auto, auto),
          inset: 10pt,
          stroke: 0.5pt + primary-color,
          fill: (x, y) => if x == 0 { primary-color.lighten(90%) },
          [*Method*], [*MSE*], [*PSNR (dB)*],
          [ADMM], [9.6e-5], [40.18],
          [RED], [7.4e-3], [21.31],
        )
        
        #rect(
          fill: accent-color.lighten(90%),
          stroke: 1pt + accent-color,
          radius: 5pt,
          inset: 7pt
        )[
          *Finding:* 
          - ADMM constraint based approach > RED penalty based regularization 
        ]
      ]
    )
]

// Task 2.2.2(c): Theoretical Analysis
#slide[
  #text(size: 28pt, weight: "bold")[Task 2.2.2(c): Is $nabla rho(x) = x - D(x)$ Correct?]
  
  #columns(2, gutter: 2em)[
    *Theoretical Requirements:*
    
    For $nabla rho(x) = x - D(x)$ to be valid, denoiser must satisfy:
    
    #rect(
      fill: primary-color.lighten(90%),
      stroke: 1pt + primary-color,
      radius: 5pt,
      inset: 10pt
    )[
      1. *Jacobian Symmetry*:
         $nabla D(x) = nabla D(x)^T$
      
      2. *Local Homogeneity*:
         $x^T nabla D(x) = D(x)$
    ]

    #colbreak()
    
    *U-Net Analysis:*
    
    ❌ *Jacobian Symmetry*: 
    - Conv layers: asymmetric operations
    - Different forward/backward paths
    
    ❌ *Local Homogeneity*:
    - ReLU: $"ReLU"(alpha x) != alpha · "ReLU"(x)$
    - Skip connections: non-linear
    
    #colbreak()
    
    *Conclusion:*
    
    #rect(
      fill: accent-color.lighten(90%),
      stroke: 1pt + accent-color,
      radius: 5pt,
      inset: 15pt
    )[
      #text(fill: red, weight: "bold")[Theoretically INCORRECT]
      
      U-Net violates both requirements!
    ]
    
    *But wait...*
    
    #rect(
      fill: primary-color.lighten(90%),
      stroke: 1pt + primary-color,
      radius: 5pt,
      inset: 15pt
    )[
      #text(fill: green.darken(20%), weight: "bold")[Practically EFFECTIVE]
      
      - Still achieves good MSE/PSNR
      - $x - D(x)$ is useful approximation
      - Works as heuristic gradient
    ]
    
    #v(0.5em)
    
    #text(fill: accent-color, weight: "bold", size: 18pt)[Theory-Practice Gap:]
    
    Deep learning often works better than theory predicts!
  ]
]

// Task 2.3: Overfitting Analysis
#slide[
  
  #text(size: 28pt, weight: "bold")[Task 2.3: Overfitting Analysis]
  
  #grid(
    
    columns: (1.8fr, 1fr),
    gutter: 2.5em,
    [
      #figure(
        image("images/task23.png", width: 100%),
      )
    ],
    [
      *Critical Finding:*
      - MSE increases after optimal point!
      
      #table(
        columns: 3,
        inset: 7pt,
        stroke: 0.5pt + primary-color,
        fill: (x, y) => if y == 0 { primary-color.lighten(90%) },
        [*Task*], [*Best*], [*Impv*],
        [Deblur], [8 iter], [+38%],
        [Inpaint], [37 iter], [+534%],
      )
      
      *Why overfitting?*
      - Denoiser accumulates errors
      - No convergence guarantee
      
      *Solution: Early stopping*
    ]
  )
]

// Module 3 Title
#slide[
  #align(center + horizon)[
    #text(size: 48pt, weight: "bold")[Module 3]
    
    
    #text(size: 32pt)[Quality Assessment & ML Pitfalls]
    
    #text(size: 20pt, fill: gray)[
      Evaluation Methodology & System Debugging
    ]
  ]
]

// Task 3.1.a: IQA Failures
#slide[
  #text(size: 28pt, weight: "bold")[Task 3.1.a: Traditional Metrics Fail]
  
  #figure(
    image("images/task31a_new.png", width: 76%),
    caption: "Six degradations exposing PSNR/SSIM failures"
  )
  
]
#slide[
  #text(size: 28pt, weight: "bold")[Task 3.1.a: Traditional Metrics Fail]
  #align(center)[
    #table(
      columns: 7,
      inset: 10pt,
      stroke: 0.5pt + primary-color,
      fill: (x, y) => if y == 0 { primary-color.lighten(90%) },
      align: center,
      [*Type*], [*PSNR*], [*SSIM*], [*FSIM*], [*VIF*], [*MS-SSIM*], [*BRISQUE*],
      [Spatial Shift], [22.92], [0.564], [0.933], [0.398], [0.822], [1.84],
      [Gaussian Noise], [24.79], [0.497], [0.944], [0.821], [0.911], [3.70],
      [Contrast], [14.21], [0.838], [0.951], [2.299], [0.833], [1.83],
      [Quantization], [40.94], [0.969], [0.994], [2.078], [0.996], [1.96],
    )
    #rect(
    fill: accent-color.lighten(90%),
    stroke: 1pt + accent-color,
    radius: 5pt,
    inset: 10pt
  )[
    #text(weight: "bold", fill: accent-color)[Finding: 26.73 dB PSNR variance for perceptually similar images!]
  ]
  ]
]

// Proposed IQA Framework
#slide[
  #text(size: 28pt, weight: "bold")[Proposed IQA Framework]
  
  #columns(2, gutter: 2em)[
    *Full-Reference (FR) Measures:*
    
    #text(fill: accent-color, weight: "bold")[FSIM]
    - Feature similarity index
    - Gradient magnitude + phase
    - Better for edges
    #colbreak()
    #text(fill: accent-color, weight: "bold")[VIF]
    - Visual information fidelity
    - Information-theoretic approach
    - Good for inpainting
    
    #text(fill: accent-color, weight: "bold")[MS-SSIM]
    - Multi-scale SSIM
    - Pyramid analysis
    - Resolution-aware
    
    #colbreak()
    
    *No-Reference (NR) Measures:*
    
    #text(fill: accent-color, weight: "bold")[BRISQUE]
    - Blind quality evaluator
    - Natural scene statistics
    - No reference needed
    #colbreak()
    #text(fill: accent-color, weight: "bold")[NIQE]
    - Natural image quality
    - Statistical regularities
    - Domain independent
    
    #text(fill: accent-color, weight: "bold")[PIQE]
    - Perception-based IQA
    - Spatial activity analysis
    - Distortion detection
  ]
  
  \
  #rect(
    fill: primary-color.lighten(90%),
    stroke: 1pt + primary-color,
    radius: 5pt,
    inset: 10pt
  )[
    *Module 2 Implication:* Use FR+NR framework for robust deblurring/inpainting evaluation
  ]
]

// Task 3.1.b: Background Paradox
#slide[
  #text(size: 28pt, weight: "bold")[Task 3.1.b: The Background Removal Paradox]
  
  #figure(
    image("images/task31a_new3.png", width: 75%),
    caption: "Background removal creates evaluation bias"
  )

]
#slide[
  #text(size: 28pt, weight: "bold")[Task 3.1.b: The Background Removal Paradox]
  #grid(
    columns: (1.4fr, 1fr),
    gutter: 2em,
    [
      #align(center+horizon)[
        #table(
          columns: 5,
          inset: 8pt,
          stroke: 0.5pt + primary-color,
          fill: (x, y) => if y == 0 { primary-color.lighten(90%) },
          [*Type*], [*PSNR Δ*], [*SSIM Δ*], [*FSIM Δ*], [*VIF Δ*],
          [Spatial], [+8.58], [+0.506], [-0.001], [+0.42],
          [Noise], [+2.96], [+0.062], [-0.080], [-1.48],
          [Contrast], [+14.15], [+0.111], [-0.006], [+0.76],
          [Gamma], [+5.83], [+0.364], [-0.013], [+0.51],
        )
      ]
    ],
    [
      *Paradox Discovery:*
      
      ✅ Traditional metrics improve:
      - PSNR: +2.96 to +14.15 dB
      - SSIM: +0.062 to +0.506
      
      ❌ Advanced metrics degrade:
      - FSIM: -0.001 to -0.080
      - VIF: up to -1.48
      
      #text(fill: red, weight: "bold")[Warning: Background removal creates systematic evaluation bias!]
    ]
  )
]

// Task 3.2.a: Original MLP
#slide[
  #text(size: 28pt, weight: "bold")[Task 3.2.a: Original MLP Analysis]
  
  #columns(2, gutter: 2em)[
    *Architecture Problems:*
    ```python
    model = nn.Sequential(
      nn.Linear(784*3, 64), # Why 3x?
      nn.Tanh(),
      nn.Linear(64, 16),    # Bottleneck!
      nn.Tanh(),
      nn.Linear(16, 10),
      nn.Softmax(dim=None)  # ERROR!
    )
    criterion = CrossEntropyLoss()
    ```
    
    *Double Softmax Bug:*
    - CrossEntropyLoss includes softmax
    - Model adds extra softmax
    - Causes gradient issues
    
    #colbreak()
    
    #figure(
      image("images/task314.png", width: 100%),
    )
    
    *Performance Issues:*
    - Loss: 1.464 (very high)
    - Class 0: 0% recall
    - Class 5: 47.8% precision
  ]
]

// Per-class Analysis
#slide[
  #text(size: 28pt, weight: "bold")[Original MLP Per-Class Performance]
  
  #table(
    columns: 6,
    inset: 10pt,
    stroke: 0.5pt + primary-color,
    fill: (x, y) => if y == 0 { primary-color.lighten(90%) },
    align: center,
    [*Class*], [*Accuracy*], [*Recall*], [*Specificity*], [*Precision*], [*Issue*],
    [0], [0.901], [0.000], [0.999], [0.000], [Complete failure],
    [1], [0.998], [0.988], [0.999], [0.994], [Good],
    [2], [0.995], [0.973], [0.997], [0.972], [Good],
    [3], [0.994], [0.970], [0.997], [0.973], [Good],
    [4], [0.996], [0.995], [0.996], [0.967], [Good],
    [5], [0.901], [0.986], [0.893], [0.478], [Over-prediction],
    [6-9], [...], [...], [...], [...], [Good],
  )
  
  *Key Problems:*
  - Catastrophic Class 0 failure (never predicted)
  - Severe Class 5 over-prediction
  - 16-unit bottleneck insufficient
  - Per-image normalization removes global patterns
]

// Task 3.2.b: Enhanced MLP
#slide[
  #text(size: 28pt, weight: "bold")[Task 3.2.b: Enhanced MLP Implementation]
  
  #grid(
    columns: (1.4fr, 1fr),
    gutter: 2em,
    [
      #figure(
        image("images/task32b.png", width: 100%),
      )
    ],
    [
      *Comprehensive Fixes:*
      1. Remove double softmax ✓
      2. Expand architecture:
         512→512→256→128→10
      3. Add BatchNorm + Dropout
      4. AdamW + LR scheduling
      5. Fix preprocessing
      
      *Results:*
      - Loss: 1.464 → 0.039
      - 97.3% reduction!
      - Val accuracy: 99.60%
      - Test accuracy: 85.33%
      
      *But Class 0 still fails!*
    ]
  )
]

// Task 3.2.c: CNN Analysis
#slide[
  #text(size: 28pt, weight: "bold")[Task 3.2.c: CNN vs MLP Comparison]
  
  #columns(2, gutter: 2em)[
    *CNN Architecture:*
    - AlexNet pretrained
    - 48.3M parameters
    - Transfer learning
    - Spatial features
    
    *Prediction:*
    Should outperform MLP due to:
    - Spatial inductive bias
    - Hierarchical features
    - Transfer learning
    
    *Reality:*
    - Test accuracy: 89.77%
    - Class 0: 22.7% → 0.2% recall
    - ImageNet→MNIST mismatch
    
    #colbreak()
    
    #figure(
      image("images/task32c.png", width: 100%),
    )
    
    #rect(
      fill: accent-color.lighten(90%),
      stroke: 1pt + accent-color,
      radius: 5pt,
      inset: 10pt
    )[
      *Finding:* 48.3M-parameter CNN failed to beat 1.6M MLP due to domain mismatch
    ]
  ]
]

// Final Comparison
#slide[
  #text(size: 28pt, weight: "bold")[Final Performance Comparison]
  
  #table(
    columns: 4,
    inset: 10pt,
    stroke: 0.5pt + primary-color,
    fill: (x, y) => if y == 0 { primary-color.lighten(90%) },
    align: (left, center, center, center),
    [*Metric*], [*Original MLP*], [*Enhanced MLP*], [*Enhanced CNN*],
    [Overall Accuracy], [~90%], [85.33%], [89.77%],
    [Training Loss], [1.464], [0.039], [0.000297],
    [Validation Loss], [1.480], [0.014], [0.005],
    [Training Efficiency], [Poor], [38.19 acc/min], [6.05 acc/min],
    [Class 0 Recall], [0.0%], [0.0%], [0.2%],
    [Parameter Count], [~150K], [1.6M], [48.3M],
  )
  
  #v(1em)
  
  #columns(3, gutter: 2em)[
    #rect(
      fill: primary-color.lighten(90%),
      stroke: 1pt + primary-color,
      radius: 5pt,
      inset: 10pt
    )[
      *Implementation > Architecture*
      
      Single-line fix yielded 97.3% improvement
    ]
    
    #rect(
      fill: primary-color.lighten(90%),
      stroke: 1pt + primary-color,
      radius: 5pt,
      inset: 10pt
    )[
      *Per-class Analysis Essential*
      
      Aggregate metrics hide catastrophic failures
    ]
    
    #rect(
      fill: primary-color.lighten(90%),
      stroke: 1pt + primary-color,
      radius: 5pt,
      inset: 10pt
    )[
      *Domain Mismatch Matters*
      
      ImageNet→MNIST transfer learning failed
    ]
  ]
]

// ML Results - Keep original slide
#slide[
  #text(size: 28pt, weight: "bold")[Systematic Improvements]
  
  
  
  #grid(
    columns: (1.4fr, 1fr),
    gutter: 2em,
    [
      #figure(
        image("images/task32b.png", width: 100%),
        caption: "97.3% loss reduction"
      )
    ],
    [
      *Single-line fix yields:*
      - Loss: 1.464 → 0.039
      - Validation: 99.60%
      
      
      *Key Lessons:*
      #enum(
        "Implementation > Architecture",
        "Per-class analysis essential",
        "Systematic debugging works"
      )
    ]
  )
]

// Conclusions
#slide[
  #text(size: 28pt, weight: "bold")[Key Achievements]
  
  
  
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 2em,
    [
      #text(weight: "bold", fill: accent-color)[Module 1]
      - 100% classification
      - Professional removal
      - Perfect detection
    ],
    [
      #text(weight: "bold", fill: accent-color)[Module 2]
      - Strong PnP-ADMM
      - Overfitting discovery
      - Early stopping
    ],
    [
      #text(weight: "bold", fill: accent-color)[Module 3]
      - IQA framework
      - Background paradox
      - 99% improvement
    ]
  )
  
  #v(1cm)
  
  #align(center)[
    #rect(fill: primary-color.lighten(90%), inset: 15pt)[
      #text(size: 18pt)[
        *Overall Impact:* Combining classical techniques with modern deep learning,
        while maintaining rigorous evaluation, yields superior results
      ]
    ]
  ]
]

// Thank You
#slide[
  #align(center + horizon)[
    #v(3cm)
    #text(size: 48pt, weight: "bold")[Thank You!]
    
    #text(size: 32pt)[Questions?]
    
    #v(2cm)

    #text(size: 24pt, fill: gray)[Bocheng Xiao (bx242)]

    #text(size: 16pt, fill: gray)[bx242\@cam.ac.uk]
  ]
  
]