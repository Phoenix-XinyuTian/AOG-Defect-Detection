# Model Architecture Diagrams (White Background)

## 1) Intensity Baseline

```mermaid
%%{init: {'theme':'base','themeVariables': {'background':'#ffffff','primaryColor':'#f5f5f5','primaryTextColor':'#111111','lineColor':'#333333'}}}%%
flowchart TB
    A[Input SEM Image] --> B[Resize to 256x256]
    B --> C[Median Blur]
    C --> D[CLAHE]
    D --> E[Global Intensity Threshold]
    E --> F[Morphological Open + Close]
    F --> G[Connected Component Filtering\n(min_area)]
    G --> H[Binary AOG Mask]
    H --> I[Overlay Generation]
    H --> J[Metric Computation\n(IoU, Dice, F1, Precision, Recall)]
```

## 2) Basic U-Net (Custom)

```mermaid
%%{init: {'theme':'base','themeVariables': {'background':'#ffffff','primaryColor':'#f5f5f5','primaryTextColor':'#111111','lineColor':'#333333'}}}%%
flowchart TB
    A[Input SEM Image] --> B[Preprocess\nResize + Median Blur + CLAHE]
    B --> C[1-channel Tensor\n1x256x256]

    subgraph UNet[Basic U-Net (custom)]
      C --> D1[Encoder Block 1\nDoubleConv]
      D1 --> P1[MaxPool]
      P1 --> D2[Encoder Block 2\nDoubleConv]
      D2 --> P2[MaxPool]
      P2 --> D3[Encoder Block 3\nDoubleConv]
      D3 --> P3[MaxPool]
      P3 --> BN[Bottleneck\nDoubleConv]
      BN --> U3[UpConv]
      U3 --> CAT3[Concat Skip 3]
      CAT3 --> DC3[DoubleConv]
      DC3 --> U2[UpConv]
      U2 --> CAT2[Concat Skip 2]
      CAT2 --> DC2[DoubleConv]
      DC2 --> U1[UpConv]
      U1 --> CAT1[Concat Skip 1]
      CAT1 --> DC1[DoubleConv]
      DC1 --> OUT[1x1 Conv\nLogits]
    end

    OUT --> S[Sigmoid]
    S --> T[Threshold (0.4)]
    T --> M[Predicted Mask]
    M --> N[Metrics + Visualization]
```

## 3) ResNet34 U-Net (SMP)

```mermaid
%%{init: {'theme':'base','themeVariables': {'background':'#ffffff','primaryColor':'#f5f5f5','primaryTextColor':'#111111','lineColor':'#333333'}}}%%
flowchart TB
    A[Input SEM Image] --> B[Preprocess\nResize + Median Blur + CLAHE\n(3 channels)]
    B --> C[Training Augment\nFlip/Rotate + Brightness + Noise + Random CLAHE]
    C --> D[Tensor\n3x256x256]

    subgraph SMPU[smp.Unet]
      D --> E[Encoder: ResNet34\n(ImageNet weights)]
      E --> F[UNet Decoder\n(skip fusion)]
      F --> G[Segmentation Head\n1 channel]
      G --> H[Sigmoid Output]
    end

    H --> I[Threshold (0.4)]
    I --> J[Binary Mask]
    J --> K[Overlay + Metrics\nIoU/Dice/F1/Precision/Recall\nArea%/Count]
```

## 4) ResNet34 U-Net++ (SMP)

```mermaid
%%{init: {'theme':'base','themeVariables': {'background':'#ffffff','primaryColor':'#f5f5f5','primaryTextColor':'#111111','lineColor':'#333333'}}}%%
flowchart TB
    A[Input SEM Image] --> B[Preprocess\nResize + Median Blur + CLAHE\n(3 channels)]
    B --> C[Training Augment\nFlip/Rotate + Brightness + Noise + Random CLAHE]
    C --> D[Tensor\n3x256x256]

    subgraph SMPUPP[smp.UnetPlusPlus]
      D --> E[Encoder: ResNet34\n(ImageNet weights)]
      E --> F[Nested Dense Skip Paths\n(UNet++ decoder)]
      F --> G[Segmentation Head\n1 channel]
      G --> H[Sigmoid Output]
    end

    H --> I[Loss (train)\nFocal + Tversky\n(alpha=0.4, beta=0.6)]
    H --> J[Threshold (e.g., 0.4)]
    J --> K[Postprocess\nConnected Component Filter\n(min_area)]
    K --> L[Binary Mask]
    L --> M[Overlay + Metrics\nIoU/Dice/F1/Precision/Recall\nArea%/Count]
```
