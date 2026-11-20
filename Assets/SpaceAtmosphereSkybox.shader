Shader "Custom/SpaceAtmosphereSkybox"
{
    Properties
    {
        _SpaceColor ("Deep Space Color", Color) = (0.008, 0.012, 0.028, 1)
        _NebulaColor ("Nebula Tint", Color) = (0.04, 0.07, 0.16, 1)
        _SkyZenithColor ("Atmosphere Zenith Color", Color) = (0.03, 0.44, 0.88, 1)
        _SkyHorizonColor ("Atmosphere Horizon Color", Color) = (0.26, 0.88, 0.98, 1)
        _SunColor ("Sun Glow Color", Color) = (1.0, 0.94, 0.82, 1)
        _StarDensity ("Star Density", Float) = 240.0
    }

    SubShader
    {
        Tags
        {
            "Queue" = "Background"
            "RenderType" = "Background"
            "PreviewType" = "Skybox"
        }
        Cull Off
        ZWrite Off

        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 dir : TEXCOORD0;
            };

            float4 _GlobalPlanetCenter;
            float _GlobalPlanetRadius;
            float _GlobalUseSphericalPlanet;

            CBUFFER_START(UnityPerMaterial)
                half4 _SpaceColor;
                half4 _NebulaColor;
                half4 _SkyZenithColor;
                half4 _SkyHorizonColor;
                half4 _SunColor;
                float _StarDensity;
            CBUFFER_END

            float Hash21(float2 p)
            {
                p = frac(p * float2(123.34, 456.21));
                p += dot(p, p + 45.32);
                return frac(p.x * p.y);
            }

            float ValueNoise2D(float2 x)
            {
                float2 i = floor(x);
                float2 f = frac(x);
                f = f * f * (3.0 - 2.0 * f);
                float a = Hash21(i);
                float b = Hash21(i + float2(1.0, 0.0));
                float c = Hash21(i + float2(0.0, 1.0));
                float d = Hash21(i + float2(1.0, 1.0));
                return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
            }

            float Hash31(float3 p)
            {
                p = frac(p * float3(443.897, 441.423, 437.195));
                p += dot(p, p.yzx + 19.19);
                return frac((p.x + p.y) * p.z);
            }

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                OUT.positionCS = TransformObjectToHClip(IN.positionOS.xyz);
                OUT.dir = IN.positionOS.xyz;
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                float3 d = normalize(IN.dir);
                Light mainLight = GetMainLight();
                float3 sunDir = normalize(mainLight.direction);

                // Deep space background with subtle galactic nebula band
                float galacticPlane = 1.0 - saturate(abs(d.y * 0.7 + d.x * 0.4) * 2.4);
                half3 spaceCol = lerp(_SpaceColor.rgb, _NebulaColor.rgb, galacticPlane * 0.55);

                // Crisp 3D procedural starfield
                float3 cell = floor(d * _StarDensity);
                float3 f = frac(d * _StarDensity) - 0.5;
                float rnd = Hash31(cell);
                if (rnd > 0.985)
                {
                    float starDist = length(f);
                    float starMask = 1.0 - smoothstep(0.04, 0.32, starDist);
                    float brightness = lerp(0.45, 1.4, frac(rnd * 137.0));
                    half3 starTint = lerp(half3(0.80, 0.88, 1.0), half3(1.0, 0.92, 0.80), frac(rnd * 53.0));
                    spaceCol += starTint * (starMask * brightness);
                }

                // Solar disc & atmospheric corona
                float sunDot = saturate(dot(d, sunDir));
                float sunCorona = pow(sunDot, 24.0) * 0.55 + pow(sunDot, 450.0) * 2.5;
                spaceCol += _SunColor.rgb * sunCorona;

                // Distant stylized banded lavender moon (visible in space and daytime atmosphere, behind clouds)
                float3 moonDir = normalize(float3(-0.34, 0.54, -0.77));
                float moonAngularRadius = 0.068;
                float moonCosLimit = cos(moonAngularRadius);
                half4 moonSample = half4(0.0, 0.0, 0.0, 0.0);
                float moonLitMix = 0.0;
                if (dot(d, moonDir) > moonCosLimit - 0.008)
                {
                    float3 moonRight = normalize(cross(float3(0.0, 1.0, 0.0), moonDir));
                    float3 moonUp = cross(moonDir, moonRight);
                    float2 moonUV = float2(dot(d, moonRight), dot(d, moonUp)) / sin(moonAngularRadius);
                    float r2 = dot(moonUV, moonUV);
                    float r = sqrt(max(r2, 1e-6));
                    float moonEdge = 1.0 - smoothstep(0.95, 1.02, r);
                    if (moonEdge > 0.001)
                    {
                        float z = sqrt(saturate(1.0 - r2));
                        float3 moonNormal = normalize(moonRight * moonUV.x + moonUp * moonUV.y + moonDir * z);
                        float bandCoord = dot(moonUV, normalize(float2(0.28, 0.96)));
                        float swirl = sin(bandCoord * 12.0 + sin(bandCoord * 5.0) * 1.4) * 0.5 + 0.5;
                        float crispBand = smoothstep(0.32, 0.68, swirl);
                        half3 moonBase = lerp(half3(0.58, 0.42, 0.76), half3(0.92, 0.88, 0.98), crispBand);
                        float moonSun = saturate(dot(moonNormal, sunDir) * 0.70 + 0.42);
                        float limbGlow = pow(saturate(1.0 - z), 2.5) * 0.30;
                        half3 moonCol = moonBase * moonSun + half3(0.78, 0.70, 0.96) * limbGlow;
                        moonSample = half4(moonCol, moonEdge);
                        moonLitMix = smoothstep(0.30, 0.85, moonSun);
                    }
                }

                // Smoothly blend into atmospheric sky dome & stylized clouds when close to the planet surface
                float atmoBlend = 0.0;
                half3 skyCol = _SkyZenithColor.rgb;
                float3 cloudUp = float3(0.0, 1.0, 0.0);
                float cloudUpDot = saturate(d.y);

                if (_GlobalUseSphericalPlanet > 0.5 && _GlobalPlanetRadius > 1.0)
                {
                    float3 toCam = _WorldSpaceCameraPos - _GlobalPlanetCenter.xyz;
                    float camDist = length(toCam);
                    float camAlt = camDist - _GlobalPlanetRadius;
                    float atmoLow = max(35.0, _GlobalPlanetRadius * 0.04);
                    float atmoHigh = max(180.0, _GlobalPlanetRadius * 0.30);
                    atmoBlend = 1.0 - smoothstep(atmoLow, atmoHigh, camAlt);

                    float3 localUp = (camDist > 1e-4) ? (toCam / camDist) : float3(0.0, 1.0, 0.0);
                    cloudUp = localUp;
                    cloudUpDot = dot(d, localUp);
                    float horizonT = pow(1.0 - saturate(cloudUpDot), 2.2);
                    skyCol = lerp(_SkyZenithColor.rgb, _SkyHorizonColor.rgb, horizonT);
                    skyCol += _SunColor.rgb * (pow(sunDot, 16.0) * 0.45 + pow(sunDot, 320.0) * 1.8);
                }

                half3 baseSky = lerp(spaceCol, skyCol, atmoBlend);

                if (moonSample.a > 0.001)
                {
                    half3 atmoTintedMoon = lerp(skyCol * 0.78 + half3(0.14, 0.12, 0.28), moonSample.rgb, moonLitMix);
                    half3 moonFinal = lerp(moonSample.rgb, atmoTintedMoon, atmoBlend);
                    baseSky = lerp(baseSky, moonFinal, moonSample.a * lerp(1.0, 0.90, atmoBlend));
                }

                if (atmoBlend > 0.001 && cloudUpDot > 0.04)
                {
                    float3 refAxis = abs(cloudUp.y) < 0.92 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
                    float3 east = normalize(cross(refAxis, cloudUp));
                    float3 north = cross(cloudUp, east);
                    float2 cloudUV = float2(dot(d, east), dot(d, north)) / max(cloudUpDot + 0.18, 0.18);

                    float c1 = ValueNoise2D(cloudUV * 1.8 + float2(_Time.y * 0.008, _Time.y * 0.004));
                    float c2 = ValueNoise2D(cloudUV * 3.9 - float2(_Time.y * 0.012, -_Time.y * 0.007));
                    float cloudField = c1 * 0.68 + c2 * 0.32;
                    float cloudMask = smoothstep(0.56, 0.61, cloudField) * smoothstep(0.04, 0.24, cloudUpDot) * atmoBlend;
                    half3 cloudCol = lerp(half3(0.74, 0.92, 0.98), half3(0.96, 0.99, 1.0), smoothstep(0.56, 0.78, cloudField));
                    baseSky = lerp(baseSky, cloudCol, cloudMask * 0.94);
                }

                return half4(baseSky, 1.0);
            }
            ENDHLSL
        }
    }
    Fallback Off
}
