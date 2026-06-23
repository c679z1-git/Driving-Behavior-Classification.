# Driving Behavior Classification

**Dataset:** https://huggingface.co/datasets/Pleumpreeti/Driving-Behavior-Classification

## Exploratory Data Analysis

<img width="1747" height="696" alt="image" src="https://github.com/user-attachments/assets/3c50c8f4-569c-40ac-ac8a-91bff3984a13" />

<img width="1575" height="562" alt="image" src="https://github.com/user-attachments/assets/4cd4ec40-eed9-426a-a51c-08b7156257af" />

## Exploratory Data Analysis Overview
**Data Overview**<br>
Every category have nearly the same number of frames between 45,000 and 46,000 Because the clip size depends directly on the number of frames, the data volume graph almost perfectly relate to the frame count. Preventing unequal model distribution.

**Duration**<br>
Swerving : Longest and varies the most. Meaning swerving is a controlled sequence with multiple steering adjustments, not just a quick steering of the wheel.<br>
Tailgating : Lasts for a moderate amount of time while feturing a few clips with extremely low duration.<br>
Normal : Have the shortest and most consistent duration.<br>

**Motion Intensity**<br>
Normal (Highest) : Driving at steady speeds makes the background scenery move within the camera quickly, causing massive pixel changes.<br>
Tailgating (Lower) : The camera locked onto the back of another car moving at the exact same speed. Because the motion between the two cars is relatively the same, only the edges of the frame show movement.<br>
Swerving (Lowest) : Swerving usually happens at lower speeds like changing lane or dodging things, resulted in less overall pixel changes.<br>
