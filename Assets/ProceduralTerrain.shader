Shader "Custom/ProceduralTerrain"
{
    Properties
    {
        _DeepOceanColor ("Deep Ocean Color", Color) = (0.02, 0.24, 0.56, 1)
        _ShallowOceanColor ("Coastal Ocean Color", Color) = (0.06, 0.56, 0.78, 1)
        _ReefColor ("Coral Reef Color", Color) = (0.14, 0.78, 0.80, 1)
        _BeachColor ("Beach Sand Color", Color) = (0.95, 0.76, 0.44, 1)
        _GrassColor ("Grassland Color", Color) = (0.16, 0.70, 0.34, 1)
        _SavannaColor ("Equatorial Savanna Color", Color) = (0.90, 0.58, 0.30, 1)
        _ForestColor ("Highland Forest Color", Color) = (0.06, 0.48, 0.38, 1)
        _RockColor ("Mountain Rock Color", Color) = (0.24, 0.50, 0.58, 1)
        _SnowColor ("Peak Snow Color", Color) = (0.92, 0.98, 1.00, 1)
        _AtmosphereColor ("Atmosphere Rim Color", Color) = (0.30, 0.80, 0.98, 1)
        _SlopeThreshold ("Slope Grass Threshold", Range(0.0, 1.0)) = 0.62
        _SlopeBlend ("Slope Blend Sharpness", Range(0.01, 0.5)) = 0.07
        _DetailScale ("Procedural Texture Scale", Float) = 0.12
        _DetailStrength ("Procedural Texture Strength", Range(0.0, 0.6)) = 0.24
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Opaque"
            "RenderPipeline" = "UniversalPipeline"
            "Queue" = "Geometry"
        }

        Pass
        {
            Name "UniversalForward"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE
            #pragma multi_compile _ _SHADOWS_SOFT

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 worldNormal : TEXCOORD1;
            };

            float4 _GlobalPlanetCenter;
            float _GlobalPlanetRadius;
            float _GlobalPlanetAmplitude;
            float _GlobalUseSphericalPlanet;
            float _GlobalCaveDepth;
            float _GlobalRenderDistance;

            CBUFFER_START(UnityPerMaterial)
                half4 _DeepOceanColor;
                half4 _ShallowOceanColor;
                half4 _ReefColor;
                half4 _BeachColor;
                half4 _GrassColor;
                half4 _SavannaColor;
                half4 _ForestColor;
                half4 _RockColor;
                half4 _SnowColor;
                half4 _AtmosphereColor;
                float _SlopeThreshold;
                float _SlopeBlend;
                float _DetailScale;
                float _DetailStrength;
            CBUFFER_END

            float Hash3D(float3 p)
            {
                p = frac(p * 0.1031);
                p += dot(p, p.yzx + 33.33);
                return frac((p.x + p.y) * p.z);
            }

            float ValueNoise3D(float3 x)
            {
                float3 i = floor(x);
                float3 f = frac(x);
                f = f * f * (3.0 - 2.0 * f);

                return lerp(
                    lerp(
                        lerp(Hash3D(i + float3(0, 0, 0)), Hash3D(i + float3(1, 0, 0)), f.x),
                        lerp(Hash3D(i + float3(0, 1, 0)), Hash3D(i + float3(1, 1, 0)), f.x),
                        f.y
                    ),
                    lerp(
                        lerp(Hash3D(i + float3(0, 0, 1)), Hash3D(i + float3(1, 0, 1)), f.x),
                        lerp(Hash3D(i + float3(0, 1, 1)), Hash3D(i + float3(1, 1, 1)), f.x),
                        f.y
                    ),
                    f.z
                );
            }

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                VertexPositionInputs posInputs = GetVertexPositionInputs(IN.positionOS.xyz);
                VertexNormalInputs normInputs = GetVertexNormalInputs(IN.normalOS);

                OUT.positionHCS = posInputs.positionCS;
                OUT.worldPos = posInputs.positionWS;
                OUT.worldNormal = normalize(normInputs.normalWS);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                float3 n = normalize(IN.worldNormal);
                float3 viewDir = normalize(GetWorldSpaceViewDir(IN.worldPos));

                float3 upDir = float3(0.0, 1.0, 0.0);
                float altitude = IN.worldPos.y - 16.0;

                if (_GlobalUseSphericalPlanet > 0.5)
                {
                    float3 radialVec = IN.worldPos - _GlobalPlanetCenter.xyz;
                    float radialDist = length(radialVec);
                    upDir = (radialDist > 1e-4) ? (radialVec / radialDist) : float3(0.0, 1.0, 0.0);
                    altitude = radialDist - _GlobalPlanetRadius;
                }

                // 3D Vector Dot Product determines terrain slope relative to local planetary gravity/up
                float rawUpDot = dot(n, upDir);
                float upDot = saturate(rawUpDot);

                // Fast 3-sample stylized procedural 3D textures (no high-frequency confetti speckles)
                float ampScale = max(_GlobalPlanetAmplitude / 32.0, 0.5);
                float macroScale = 1.0 / max(sqrt(ampScale), 1.0);
                float macroBiome = ValueNoise3D(IN.worldPos * (_DetailScale * 0.035 * macroScale));
                float midPatch = ValueNoise3D(IN.worldPos * (_DetailScale * 0.16 * macroScale));
                float rockDetail = ValueNoise3D(IN.worldPos * (_DetailScale * 0.85 * macroScale));
                float fineGrain = rockDetail;

                // Precalculus sin(latitude) = |upDir.y| for polar ice caps and equatorial savanna zones
                float latitude = saturate(abs(upDir.y) + (macroBiome - 0.5) * 0.14);
                float polarFactor = smoothstep(0.68, 0.86, latitude);
                float equatorFactor = 1.0 - smoothstep(0.10, 0.32, latitude);

                float slopeThresholdVar = _SlopeThreshold + (midPatch - 0.5) * 0.12;
                float grassMask = smoothstep(slopeThresholdVar - _SlopeBlend, slopeThresholdVar + _SlopeBlend, upDot);

                // Stylized emerald/teal meadows with broad terracotta-orange valley dirt patches
                float forestMask = saturate(smoothstep(4.5 * ampScale, 11.5 * ampScale, altitude) * 0.72 + smoothstep(0.48, 0.70, macroBiome) * 0.45);
                half3 temperateVeg = lerp(_GrassColor.rgb, _ForestColor.rgb, forestMask);
                float valleyMask = 1.0 - smoothstep(1.5 * ampScale, 7.5 * ampScale, altitude);
                float dirtField = macroBiome * 0.45 + midPatch * 0.55;
                float dirtPatchMask = smoothstep(0.56, 0.67, dirtField) * saturate(valleyMask * 0.85 + equatorFactor * 0.45);
                half3 vegCol = lerp(temperateVeg, _SavannaColor.rgb, dirtPatchMask);
                vegCol *= (0.96 + (midPatch - 0.5) * _DetailStrength);

                // Cool glacial teal-slate mountain cliffs + subterranean strata
                float rockPattern = (rockDetail * 0.60 + fineGrain * 0.40) - 0.5;
                half3 baseRock = lerp(_RockColor.rgb, half3(0.44, 0.56, 0.60), equatorFactor * 0.35);
                half3 rockCol = baseRock * (0.94 + rockPattern * (_DetailStrength * 1.4));

                float oceanMask = 1.0 - smoothstep(-0.35, 0.08, altitude);
                float oceanSlopeGate = smoothstep(0.35, 0.65, upDot);
                float oceanDepthGate = 1.0 - smoothstep(3.5, 8.0, -altitude);
                oceanMask *= oceanSlopeGate * oceanDepthGate;

                float caveCeilingMask = 1.0 - smoothstep(-0.35, -0.05, rawUpDot);
                float subSurfaceMask = (_GlobalCaveDepth > 0.5) ? (smoothstep(6.0, 18.0, -altitude) * (1.0 - oceanMask)) : 0.0;
                float caveRockMask = saturate(max(caveCeilingMask, subSurfaceMask));
                float strata = sin(altitude * 0.45 + rockDetail * 5.0) * 0.5 + 0.5;
                half3 caveRockCol = half3(0.42, 0.34, 0.30) * (0.90 + strata * 0.25);
                rockCol = lerp(rockCol, caveRockCol, caveRockMask * 0.65);

                float highRockMask = smoothstep(9.5 * ampScale, 15.0 * ampScale, altitude);
                half3 landAlbedo = lerp(rockCol, vegCol, grassMask * (1.0 - highRockMask * 0.85));

                // Tight shoreline beach sand strictly along flat shores
                float beachTop = min(14.0, 0.7 * ampScale);
                float beachMask = (1.0 - smoothstep(0.0, beachTop, altitude)) * smoothstep(-0.4, 0.2, altitude) * smoothstep(0.75, 0.90, upDot);
                half3 beachCol = _BeachColor.rgb * (1.0 + (fineGrain - 0.5) * _DetailStrength * 0.8);
                landAlbedo = lerp(landAlbedo, beachCol, beachMask);

                // Crisp alpine & polar snow peaks
                float baseSnowHeight = (14.0 + (macroBiome - 0.5) * 2.5) * ampScale;
                float snowHeight = lerp(baseSnowHeight, -0.5, polarFactor);
                float snowMask = smoothstep(snowHeight, snowHeight + 3.8 * ampScale, altitude) * smoothstep(0.28, 0.62, upDot);
                half3 snowCol = _SnowColor.rgb * (0.95 + fineGrain * 0.08);
                landAlbedo = lerp(landAlbedo, snowCol, snowMask);

                // Spherical ocean with shallow turquoise coral reefs, polar sea ice, and animated surf
                float depthT = smoothstep(0.15, 1.55, -altitude + (macroBiome - 0.5) * 0.35);
                float reefMask = (1.0 - smoothstep(0.10, 0.95, -altitude)) * (1.0 - polarFactor) * smoothstep(0.10, 0.45, equatorFactor + macroBiome * 0.4);
                half3 oceanCol = lerp(_ShallowOceanColor.rgb, _DeepOceanColor.rgb, depthT);
                oceanCol = lerp(oceanCol, _ReefColor.rgb, reefMask * 0.65);

                // Polar sea ice shelves at high latitudes
                float seaIceMask = smoothstep(0.78, 0.90, latitude + (midPatch - 0.5) * 0.10);
                oceanCol = lerp(oceanCol, half3(0.88, 0.93, 0.98), seaIceMask * 0.85);

                float surfWave = sin(altitude * 10.0 - _Time.y * 2.5) * 0.5 + 0.5;
                float foamMask = smoothstep(-0.38, -0.04, altitude) * (1.0 - smoothstep(-0.04, 0.18, altitude));
                oceanCol = lerp(oceanCol, half3(0.84, 0.93, 0.98), foamMask * (0.35 + 0.45 * surfWave) * (0.6 + fineGrain * 0.4));

                half3 baseAlbedo = lerp(landAlbedo, oceanCol, oceanMask);

                // Smooth geometric normals across land; 100% smooth spherical upDir + gentle waves across lakes/ocean
                float3 shadingNormal = n;
                if (oceanMask > 0.01)
                {
                    float3 oceanNormal = upDir;
                    if (seaIceMask < 0.95)
                    {
                        float waveMask = smoothstep(0.01, 0.25, oceanMask) * (1.0 - seaIceMask);
                        float3 p1 = IN.worldPos * 0.32 + float3(_Time.y * 0.85, _Time.y * 0.40, -_Time.y * 0.65);
                        float3 p2 = IN.worldPos * 0.68 + float3(-_Time.y * 0.60, _Time.y * 0.75, _Time.y * 0.50);
                        float3 grad1 = float3(
                            cos(p1.x + 0.7 * sin(p1.z)),
                            cos(p1.y + 0.7 * sin(p1.x)),
                            cos(p1.z + 0.7 * sin(p1.y))
                        );
                        float3 grad2 = float3(
                            cos(p2.z - 0.6 * sin(p2.y)),
                            cos(p2.x - 0.6 * sin(p2.z)),
                            cos(p2.y - 0.6 * sin(p2.x))
                        );
                        float3 waveGrad = grad1 * 0.040 + grad2 * 0.024;
                        float3 tangentWave = waveGrad - upDir * dot(waveGrad, upDir);
                        oceanNormal = normalize(upDir - tangentWave * waveMask);
                    }
                    shadingNormal = normalize(lerp(n, oceanNormal, oceanMask));
                }

                // Main directional sunlight + water specular + altitude-aware rim & subtle horizon haze
                float4 shadowCoord = TransformWorldToShadowCoord(IN.worldPos);
                Light mainLight = GetMainLight(shadowCoord);
                float ndotl = saturate(dot(shadingNormal, mainLight.direction));
                half3 directLight = mainLight.color * (ndotl * mainLight.shadowAttenuation * 1.12);

                float3 halfVec = normalize(mainLight.direction + viewDir);
                float specPower = lerp(14.0, 96.0, oceanMask * (1.0 - seaIceMask));
                float specStrength = lerp(0.04, 0.58, oceanMask * (1.0 - seaIceMask));
                float spec = pow(saturate(dot(shadingNormal, halfVec)), specPower) * specStrength * ndotl * mainLight.shadowAttenuation;

                half3 skyBounce = lerp(half3(0.10, 0.24, 0.38), half3(0.22, 0.42, 0.54), saturate(dot(shadingNormal, upDir) * 0.5 + 0.5));
                half3 skyAmbient = SampleSH(shadingNormal) * 0.38 + skyBounce * 0.72;
                skyAmbient = max(skyAmbient, half3(0.06, 0.07, 0.09) * caveRockMask);
                half3 litColor = baseAlbedo * (directLight + skyAmbient) + mainLight.color * spec;

                float safeRadius = max(_GlobalPlanetRadius, 1.0);
                float camAlt = (_GlobalUseSphericalPlanet > 0.5)
                    ? (length(_WorldSpaceCameraPos - _GlobalPlanetCenter.xyz) - safeRadius)
                    : 0.0;
                float spaceRimFactor = (_GlobalUseSphericalPlanet > 0.5)
                    ? smoothstep(safeRadius * 0.05, safeRadius * 0.25, camAlt)
                    : 0.0;
                float fresnel = pow(1.0 - saturate(dot(shadingNormal, viewDir)), 4.2);
                float daySide = saturate(dot(upDir, mainLight.direction) * 0.65 + 0.35);
                litColor += _AtmosphereColor.rgb * (fresnel * lerp(0.10, 0.40, spaceRimFactor) * daySide);

                if (_GlobalRenderDistance > 100.0 && _GlobalUseSphericalPlanet > 0.5)
                {
                    float fogPresence = 1.0 - smoothstep(safeRadius * 0.12, safeRadius * 0.30, camAlt);
                    float camDistToFrag = length(_WorldSpaceCameraPos - IN.worldPos);
                    float fogT = smoothstep(_GlobalRenderDistance * 0.58, _GlobalRenderDistance * 1.00, camDistToFrag) * fogPresence * 0.68;
                    half3 horizonFogCol = lerp(half3(0.04, 0.08, 0.16), _AtmosphereColor.rgb * 1.05, daySide);
                    litColor = lerp(litColor, horizonFogCol, saturate(fogT));
                }

                return half4(litColor, 1.0);
            }
            ENDHLSL
        }

        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }

            ZWrite On
            ZTest LEqual

            HLSLPROGRAM
            #pragma vertex ShadowPassVertex
            #pragma fragment ShadowPassFragment

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"

            float3 _LightDirection;

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
            };

            Varyings ShadowPassVertex(Attributes input)
            {
                Varyings output;
                float3 positionWS = TransformObjectToWorld(input.positionOS.xyz);
                float3 normalWS = TransformObjectToWorldNormal(input.normalOS);
                output.positionCS = TransformWorldToHClip(ApplyShadowBias(positionWS, normalWS, _LightDirection));
                return output;
            }

            half4 ShadowPassFragment(Varyings input) : SV_Target
            {
                return 0;
            }
            ENDHLSL
        }
    }
}
