using System.Collections.Generic;
using UnityEngine;

public class TerraformingCamera : MonoBehaviour
{
	Vector3 _hitPoint;
	Camera _cam;

	public float BrushSize = 24f;
	public float SculptInterval = 0.06f;
	public float ColliderBakeInterval = 0.08f;
	public PlanetOctree Octree;

	[Header("Camera Navigation")]
	public float MoveSpeed = 70f;
	public float TurnSpeed = 90f;
	public float MouseSensitivity = 2.2f;
	public bool LockMouseOnStart = true;
	public bool AutoAlignToPlanet = true;
	public float MouseLookSmoothing = 18f;
	public float SpeedMultiplier = 1f;

	readonly Collider[] _overlapColliders = new Collider[16];
	readonly HashSet<Chunk> _sculptedChunks = new HashSet<Chunk>();
	readonly List<Chunk> _sculptedChunkList = new List<Chunk>(16);
	readonly HashSet<Chunk> _pendingColliderChunks = new HashSet<Chunk>();
	float _nextColliderBakeTime;
	float _nextSculptTime;
	float _nextHudRefreshTime;
	float _smoothedFrameTimeMs;
	float _currentFlightSpeed;
	float _smoothedMouseX;
	float _smoothedMouseY;
	bool _showHud = true;
	bool _hudDirty = true;
	bool _cursorLockedFlight;
	string _cachedHudText = string.Empty;
	GUIStyle _hudBoxStyle;
	GUIStyle _hudLabelStyle;
	Texture2D _hudBgTexture;
	Texture2D _crosshairTexture;

	private void Awake() {
		_cam = GetComponent<Camera>();
		if (Octree == null) {
			Octree = FindFirstObjectByType<PlanetOctree>();
		}
	}

	private void Start() {
		if (LockMouseOnStart) {
			SetCursorLock(true);
		}
	}

	private void OnDestroy() {
		if (_cursorLockedFlight) {
			SetCursorLock(false);
		}
		if (_hudBgTexture != null) {
			Destroy(_hudBgTexture);
			_hudBgTexture = null;
		}
		if (_crosshairTexture != null) {
			Destroy(_crosshairTexture);
			_crosshairTexture = null;
		}
	}

	private void Update() {
		float frameMs = Time.unscaledDeltaTime * 1000f;
		_smoothedFrameTimeMs = _smoothedFrameTimeMs <= 0f
			? frameMs
			: Mathf.Lerp(_smoothedFrameTimeMs, frameMs, 0.1f);

		HandleDemoHotkeys();
		HandleNavigation();

		if (_cam != null) {
			float safeAlt = Mathf.Max(0f, ComputeCameraAltitude());
			_cam.nearClipPlane = Mathf.Clamp(safeAlt * 0.002f, 0.5f, 45f);
		}

		if (_showHud && (_hudDirty || Time.unscaledTime >= _nextHudRefreshTime)) {
			RefreshHudText();
			_hudDirty = false;
			_nextHudRefreshTime = Time.unscaledTime + 0.15f;
		}
	}

	private void LateUpdate() {
		if (Time.time >= _nextSculptTime) {
			if (Input.GetMouseButton(0)) {
				Terraform(true);
				_nextSculptTime = Time.time + SculptInterval;
			}
			else if (Input.GetMouseButton(1)) {
				Terraform(false);
				_nextSculptTime = Time.time + SculptInterval;
			}
		}

		bool mouseReleased = Input.GetMouseButtonUp(0) || Input.GetMouseButtonUp(1);
		if (_pendingColliderChunks.Count > 0 && (mouseReleased || Time.time >= _nextColliderBakeTime)) {
			foreach (Chunk chunk in _pendingColliderChunks) {
				if (chunk != null && chunk.gameObject.activeInHierarchy) {
					chunk.FlushCollider();
				}
			}
			_pendingColliderChunks.Clear();
			_nextColliderBakeTime = Time.time + ColliderBakeInterval;
		}
	}

	private void HandleDemoHotkeys() {
		if (Input.GetKeyDown(KeyCode.H)) {
			_showHud = !_showHud;
			if (_showHud) {
				_hudDirty = true;
			}
		}

		if (Input.GetKeyDown(KeyCode.C) || Input.GetKeyDown(KeyCode.Tab)) {
			SetCursorLock(!_cursorLockedFlight);
			_hudDirty = true;
		}
		else if (_cursorLockedFlight && Input.GetKeyDown(KeyCode.Escape)) {
			SetCursorLock(false);
			_hudDirty = true;
		}
		else if (!_cursorLockedFlight && Input.GetMouseButtonDown(2)) {
			SetCursorLock(true);
			_hudDirty = true;
		}

		if (Input.GetKeyDown(KeyCode.LeftBracket)) {
			BrushSize = Mathf.Clamp(BrushSize - 4f, 4f, 300f);
			_hudDirty = true;
		}
		if (Input.GetKeyDown(KeyCode.RightBracket)) {
			BrushSize = Mathf.Clamp(BrushSize + 4f, 4f, 300f);
			_hudDirty = true;
		}

		if (Octree == null) {
			return;
		}

		if (Input.GetKeyDown(KeyCode.F)) {
			Octree.SetSmoothShading(!Octree.SmoothShadingEnabled);
			_hudDirty = true;
		}
		if (Input.GetKeyDown(KeyCode.G)) {
			Octree.ShowOctreeBounds = !Octree.ShowOctreeBounds;
			_hudDirty = true;
		}
		if (Input.GetKeyDown(KeyCode.N)) {
			Octree.ToggleNoiseType();
			_hudDirty = true;
		}
		if (Input.GetKeyDown(KeyCode.L)) {
			Octree.FreezeStreaming = !Octree.FreezeStreaming;
			_hudDirty = true;
		}
		if (Input.GetKeyDown(KeyCode.K)) {
			Octree.TogglePartialCulling();
			_hudDirty = true;
		}
		if (Input.GetKeyDown(KeyCode.Minus)) {
			Octree.SurfaceRenderDistance = Mathf.Max(2000f, Octree.SurfaceRenderDistance - 1000f);
			Octree.RefreshOctree();
			_hudDirty = true;
		}
		if (Input.GetKeyDown(KeyCode.Equals)) {
			Octree.SurfaceRenderDistance = Mathf.Min(20000f, Octree.SurfaceRenderDistance + 1000f);
			Octree.RefreshOctree();
			_hudDirty = true;
		}
		if (Input.GetKeyDown(KeyCode.R)) {
			Octree.ClearAllStrokesAndRebuild();
			_hudDirty = true;
		}
	}

	private float ComputeCameraAltitude() {
		if (Octree != null && Octree.UseSphericalPlanet) {
			return Vector3.Distance(transform.position, Octree.PlanetCenter) - Octree.PlanetRadius;
		}
		return transform.position.y - GridMetrics.GroundLevel;
	}

	private void SetCursorLock(bool locked) {
		_cursorLockedFlight = locked;
		Cursor.lockState = locked ? CursorLockMode.Locked : CursorLockMode.None;
		Cursor.visible = !locked;
	}

	private void HandleNavigation() {
		float scroll = Input.mouseScrollDelta.y;
		if (Mathf.Abs(scroll) > 0.01f) {
			if (Input.GetKey(KeyCode.LeftAlt)) {
				BrushSize = Mathf.Clamp(BrushSize + scroll * 3f, 4f, 300f);
			}
			else {
				SpeedMultiplier = Mathf.Clamp(SpeedMultiplier * (scroll > 0f ? 1.15f : 1f / 1.15f), 0.2f, 6.0f);
			}
			_hudDirty = true;
		}

		float altitude = ComputeCameraAltitude();
		float safeAlt = Mathf.Max(0f, altitude);
		float altitudeFactor = 1f + Mathf.Clamp(safeAlt / 120f, 0f, 170f);
		float baseSpeed = MoveSpeed * altitudeFactor * SpeedMultiplier;
		float speed = Input.GetKey(KeyCode.LeftShift) ? baseSpeed * 3.0f : baseSpeed;
		_currentFlightSpeed = speed;

		Vector3 move = Vector3.zero;
		if (Input.GetKey(KeyCode.W)) move += transform.forward;
		if (Input.GetKey(KeyCode.S)) move -= transform.forward;
		if (Input.GetKey(KeyCode.D)) move += transform.right;
		if (Input.GetKey(KeyCode.A)) move -= transform.right;
		if (Input.GetKey(KeyCode.E) || Input.GetKey(KeyCode.Space)) move += transform.up;
		if (Input.GetKey(KeyCode.Q) || Input.GetKey(KeyCode.LeftControl)) move -= transform.up;

		if (move.sqrMagnitude > 0f) {
			transform.position += move.normalized * (speed * Time.deltaTime);
		}

		float yaw = 0f;
		float pitch = 0f;
		if (Input.GetKey(KeyCode.RightArrow)) yaw += 1f;
		if (Input.GetKey(KeyCode.LeftArrow)) yaw -= 1f;
		if (Input.GetKey(KeyCode.UpArrow)) pitch -= 1f;
		if (Input.GetKey(KeyCode.DownArrow)) pitch += 1f;

		if (yaw != 0f || pitch != 0f) {
			transform.Rotate(Vector3.up, yaw * TurnSpeed * Time.deltaTime, Space.Self);
			transform.Rotate(Vector3.right, pitch * TurnSpeed * Time.deltaTime, Space.Self);
		}

		bool mouseLookActive = _cursorLockedFlight || Input.GetMouseButton(2) || Input.GetKey(KeyCode.LeftAlt);
		float targetMouseX = mouseLookActive ? Input.GetAxisRaw("Mouse X") * MouseSensitivity : 0f;
		float targetMouseY = mouseLookActive ? Input.GetAxisRaw("Mouse Y") * MouseSensitivity : 0f;
		float lookLerp = 1f - Mathf.Exp(-Mathf.Max(1f, MouseLookSmoothing) * Time.unscaledDeltaTime);
		_smoothedMouseX = Mathf.Lerp(_smoothedMouseX, targetMouseX, lookLerp);
		_smoothedMouseY = Mathf.Lerp(_smoothedMouseY, targetMouseY, lookLerp);

		if (Mathf.Abs(_smoothedMouseX) > 1e-4f || Mathf.Abs(_smoothedMouseY) > 1e-4f) {
			transform.Rotate(Vector3.up, _smoothedMouseX, Space.Self);
			transform.Rotate(Vector3.right, -_smoothedMouseY, Space.Self);
		}

		if (AutoAlignToPlanet && Octree != null && Octree.UseSphericalPlanet && Octree.PlanetRadius > 1f) {
			float maxAlignAlt = Octree.PlanetRadius * 0.35f;
			if (safeAlt < maxAlignAlt) {
				Vector3 radialVec = transform.position - Octree.PlanetCenter;
				if (radialVec.sqrMagnitude > 1e-4f) {
					Vector3 radialUp = radialVec.normalized;
					float absFwdUp = Mathf.Abs(Vector3.Dot(transform.forward, radialUp));
					if (absFwdUp < 0.85f) {
						float altFade = 1f - Mathf.InverseLerp(Octree.PlanetRadius * 0.25f, maxAlignAlt, safeAlt);
						float viewFade = 1f - Mathf.InverseLerp(0.70f, 0.85f, absFwdUp);
						float alignRate = 1f - Mathf.Exp(-4.5f * altFade * viewFade * Time.unscaledDeltaTime);
						Quaternion targetRot = Quaternion.LookRotation(transform.forward, radialUp);
						transform.rotation = Quaternion.Slerp(transform.rotation, targetRot, alignRate);
					}
				}
			}
		}
	}

	private void Terraform(bool add) {
		Vector3 aimScreenPos = _cursorLockedFlight
			? new Vector3(Screen.width * 0.5f, Screen.height * 0.5f, 0f)
			: Input.mousePosition;
		Ray ray = _cam.ScreenPointToRay(aimScreenPos);
		if (Octree != null && Octree.EnableOctreeMode) {
			if (Octree.RaycastTerrain(ray, 12000f, out Vector3 hitPoint)) {
				_hitPoint = hitPoint;
				HashSet<Chunk> colliderSink = Octree.AutoBakeNearbyColliders ? _pendingColliderChunks : null;
				Octree.ApplySculptAtWorldPoint(_hitPoint, BrushSize, add, colliderSink);
			}
			return;
		}

		RaycastHit hit;

		if (Physics.Raycast(ray, out hit, 200000f)) {
			_hitPoint = hit.point;

			_sculptedChunks.Clear();
			_sculptedChunkList.Clear();

			// Query all chunk colliders touched by the spherical brush in standalone non-octree mode
			int hitCount = Physics.OverlapSphereNonAlloc(_hitPoint, BrushSize, _overlapColliders);
			for (int i = 0; i < hitCount; i++) {
				Chunk chunk = _overlapColliders[i].GetComponent<Chunk>();
				if (chunk != null && _sculptedChunks.Add(chunk)) {
					_sculptedChunkList.Add(chunk);
				}
			}

			if (_sculptedChunkList.Count == 0) {
				Chunk hitChunk = hit.collider.GetComponent<Chunk>();
				if (hitChunk != null && _sculptedChunks.Add(hitChunk)) {
					_sculptedChunkList.Add(hitChunk);
				}
			}

			for (int i = 0; i < _sculptedChunkList.Count; i++) {
				Chunk chunk = _sculptedChunkList[i];
				chunk.EditWeights(_hitPoint, BrushSize, add, updateCollider: false);
				_pendingColliderChunks.Add(chunk);
			}
		}
	}

	static string FormatMeters(float meters) {
		return Mathf.Abs(meters) >= 1000f ? $"{meters / 1000f:F2}km" : $"{meters:F0}m";
	}

	static string FormatSpeed(float metersPerSec) {
		return Mathf.Abs(metersPerSec) >= 1000f ? $"{metersPerSec / 1000f:F2} km/s" : $"{metersPerSec:F0} m/s";
	}

	private void RefreshHudText() {
		float fps = _smoothedFrameTimeMs > 0.01f ? 1000f / _smoothedFrameTimeMs : 0f;
		float alt = ComputeCameraAltitude();
		float planetRadius = Octree != null ? Octree.PlanetRadius : 0f;
		float flightSpeed = _currentFlightSpeed > 0f ? _currentFlightSpeed : MoveSpeed;
		string mouseMode = _cursorLockedFlight ? "LOCKED" : "FREE";
		string octreeInfo = "Octree: N/A";
		string meshInfo = "Mesh: N/A";
		string modeInfo = $"Mouse [Tab/Esc]: {mouseMode} | Throttle [Scroll]: {SpeedMultiplier:F1}x | Brush [[/]]: {BrushSize:F0}m";

		if (Octree != null) {
			Octree.GetMeshTelemetry(
				out int tris,
				out int rawVerts,
				out int weldedVerts,
				out float savedPct,
				out int lruCached
			);
			string shadingMode = Octree.SmoothShadingEnabled ? "Smooth Welded" : "Flat Faceted";
			string noiseMode = Octree.NoiseGenerator != null && Octree.NoiseGenerator.UsePerlinNoise ? "Perlin" : "OpenSimplex2";
			string wireMode = Octree.ShowOctreeBounds ? "ON" : "OFF";
			string streamMode = Octree.FreezeStreaming ? "FROZEN" : "LIVE";
			string cullMode = Octree.PartialCullingEnabled ? "ON" : "OFF";

			octreeInfo = $"Active: {Octree.ActiveChunkCount} | Pooled: {Octree.PooledChunkCount} | Empty: {Octree.EmptyLeafCount} | Culled (Shell/Horizon/Frustum): {Octree.CulledShellCount}/{Octree.CulledHorizonCount}/{Octree.CulledFrustumCount} | Queue: {Octree.PendingBuildCount} | Async GPU: {Chunk.ActiveAsyncReadbackCount}/{Chunk.MaxAsyncReadbackSlots}";
			meshInfo = $"Tris: {tris:N0} | Verts: {weldedVerts:N0} / {rawVerts:N0} ({savedPct:F1}% saved) | LRU: {lruCached}/8 | Strokes: {Octree.RecordedStrokeCount}";
			modeInfo = $"Mouse [Tab]: {mouseMode} | Throttle [Scroll]: {SpeedMultiplier:F1}x | View [-/=]: {FormatMeters(Octree.SurfaceRenderDistance)} | Brush [[/]]: {BrushSize:F0}m | Shade [F]: {shadingMode} | Wire [G]: {wireMode} | Stream [L]: {streamMode} | Cull [K]: {cullMode} | {noiseMode} [N]";
		}

		const string controlsLine = "Fly: WASD+Space/Ctrl+Shift | Steer: Mouse | Sculpt: LMB/RMB | Reset [R] | HUD [H]";
		_cachedHudText = $"60km GPU Marching Cubes Planet (R={FormatMeters(planetRadius)})  —  {fps:F0} FPS ({_smoothedFrameTimeMs:F1} ms) | Alt: {FormatMeters(alt)} | Speed: {FormatSpeed(flightSpeed)}\n{octreeInfo}\n{meshInfo}\n{modeInfo}\n{controlsLine}";
	}

	private void OnGUI() {
		EnsureHudStyles();

		if (_cursorLockedFlight && _crosshairTexture != null) {
			float cx = Screen.width * 0.5f;
			float cy = Screen.height * 0.5f;
			Color prev = GUI.color;
			GUI.color = new Color(1f, 1f, 1f, 0.85f);
			GUI.DrawTexture(new Rect(cx - 7f, cy - 1f, 14f, 2f), _crosshairTexture);
			GUI.DrawTexture(new Rect(cx - 1f, cy - 7f, 2f, 14f), _crosshairTexture);
			GUI.color = prev;
		}

		if (!_showHud) {
			return;
		}

		if (string.IsNullOrEmpty(_cachedHudText)) {
			RefreshHudText();
		}

		GUILayout.BeginArea(new Rect(14f, 14f, 1020f, 138f), _hudBoxStyle);
		GUILayout.Label(_cachedHudText, _hudLabelStyle);
		GUILayout.EndArea();
	}

	private void EnsureHudStyles() {
		if (_hudBgTexture == null) {
			_hudBgTexture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
			_hudBgTexture.SetPixel(0, 0, new Color(0.05f, 0.08f, 0.12f, 0.78f));
			_hudBgTexture.Apply();
		}
		if (_crosshairTexture == null) {
			_crosshairTexture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
			_crosshairTexture.SetPixel(0, 0, Color.white);
			_crosshairTexture.Apply();
		}

		if (_hudBoxStyle == null) {
			_hudBoxStyle = new GUIStyle(GUI.skin.box) {
				padding = new RectOffset(10, 10, 8, 8)
			};
			_hudBoxStyle.normal.background = _hudBgTexture;
		}

		if (_hudLabelStyle == null) {
			_hudLabelStyle = new GUIStyle(GUI.skin.label) {
				fontSize = 12,
				richText = false,
				wordWrap = false
			};
			_hudLabelStyle.normal.textColor = new Color(0.93f, 0.96f, 1f, 1f);
		}
	}

	private void OnDrawGizmos() {
		Gizmos.color = Color.yellow;
		Gizmos.DrawWireSphere(_hitPoint, BrushSize);
	}
}
