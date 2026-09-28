[CmdletBinding()]
param()

# Incremental extension build against the installed, pinned engine objects.
# The existing bridge and manifest remain untouched for historical evaluation.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$baseRoot = Join-Path $projectRoot ".simulator\sts_lightspeed"
$manifestPath = Join-Path $baseRoot "build_manifest.json"
$manifest = Get-Content -Raw -Encoding UTF8 -LiteralPath $manifestPath | ConvertFrom-Json
$outputDir = Join-Path $projectRoot "outputs\card-selection"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$bridgeSource = Join-Path $projectRoot "tools\sts_lightspeed\decision_bridge.cpp"
$objectPath = Join-Path $outputDir "bridge.o"
$executable = Join-Path $outputDir "sts_lightspeed_bridge.exe"
$started = Get-Date

# Match the maintained base builder's source list; do not link stale objects
# left over from earlier engine builds (especially the unpatched searcher).
$sourceDir = Join-Path $baseRoot "source"
$objectsDir = Join-Path $baseRoot "build\obj"
$actionsPatch = Join-Path $projectRoot "tools\sts_lightspeed\actions_upgrade_hand.patch"
$overlayRoot = Join-Path $outputDir "source_overlay"
$actionsSource = Join-Path $overlayRoot "src\combat\Actions.cpp"
$actionsObject = Join-Path $outputDir "actions.o"
$ragePatch = Join-Path $projectRoot "tools\sts_lightspeed\cards_rage_cost.patch"
$cardsHeader = Join-Path $overlayRoot "include\constants\Cards.h"
$cardObject = Join-Path $outputDir "card_instance.o"
$mechanicsPatch = Join-Path $projectRoot "tools\sts_lightspeed\card_mechanics.patch"
$cardSource = Join-Path $overlayRoot "src\combat\CardInstance.cpp"
$battleSource = Join-Path $overlayRoot "src\combat\BattleContext.cpp"
$battleObject = Join-Path $outputDir "battle_context.o"
$memoryPatch = Join-Path $projectRoot "tools\sts_lightspeed\public_draw_memory.patch"
$memoryHeader = Join-Path $projectRoot "tools\sts_lightspeed\public_draw_memory.h"
$managerSource = Join-Path $overlayRoot "src\combat\CardManager.cpp"
$managerObject = Join-Path $outputDir "card_manager.o"
New-Item -ItemType Directory -Force -Path (Split-Path $actionsSource -Parent) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $cardsHeader -Parent) | Out-Null
Copy-Item -LiteralPath (Join-Path $sourceDir "src\combat\Actions.cpp") -Destination $actionsSource
Copy-Item -LiteralPath (Join-Path $sourceDir "src\combat\CardInstance.cpp") -Destination $cardSource
Copy-Item -LiteralPath (Join-Path $sourceDir "src\combat\BattleContext.cpp") -Destination $battleSource
Copy-Item -LiteralPath (Join-Path $sourceDir "src\combat\CardManager.cpp") -Destination $managerSource
# Recreate the retained exhaust overlay from declared source and tracked patches.
Copy-Item -LiteralPath (Join-Path $sourceDir "include\constants\Cards.h") -Destination $cardsHeader
Push-Location $projectRoot
try {
    & git apply --directory=outputs/card-selection/source_overlay $actionsPatch
    if ($LASTEXITCODE -ne 0) { throw "Hand upgrade patch failed" }
    & git apply --directory=outputs/card-selection/source_overlay (Join-Path $projectRoot "tools\sts_lightspeed\cards_seeing_red.patch")
    if ($LASTEXITCODE -ne 0) { throw "Retained card exhaust patch failed" }
    & git apply --directory=outputs/card-selection/source_overlay $ragePatch
    if ($LASTEXITCODE -ne 0) { throw "Rage cost patch failed" }
    & git apply --directory=outputs/card-selection/source_overlay $mechanicsPatch
    if ($LASTEXITCODE -ne 0) { throw "Card mechanics patch failed" }
    & git apply --directory=outputs/card-selection/source_overlay $memoryPatch
    if ($LASTEXITCODE -ne 0) { throw "Public draw memory patch failed" }
} finally { Pop-Location }
$compileFlags = @("-DSTS_PUBLIC_DRAW_MEMORY", "-I", (Join-Path $projectRoot "tools\sts_lightspeed"), "-DSTS_CORRECTED_CARD_MECHANICS", "-I", (Join-Path $overlayRoot "include")) + @($manifest.compiler.compile_flags)
& $manifest.compiler.path @compileFlags -c $actionsSource -o $actionsObject
if ($LASTEXITCODE -ne 0) { throw "Hand upgrade action compilation failed" }
# CardInstance owns both runtime consumers of getEnergyCost: construction and upgrade.
& $manifest.compiler.path @compileFlags -c $cardSource -o $cardObject
if ($LASTEXITCODE -ne 0) { throw "Card cost compilation failed" }
& $manifest.compiler.path @compileFlags -c $battleSource -o $battleObject
if ($LASTEXITCODE -ne 0) { throw "Battle card mechanics compilation failed" }
& $manifest.compiler.path @compileFlags -c $managerSource -o $managerObject
if ($LASTEXITCODE -ne 0) { throw "Public draw memory compilation failed" }
$engineObjects = @(
    Get-ChildItem -LiteralPath (Join-Path $sourceDir "src") -Recurse -Filter "*.cpp" |
        Where-Object { $_.FullName -ne (Join-Path $sourceDir "src\sim\search\BattleScumSearcher2.cpp") } |
        Sort-Object FullName | ForEach-Object {
            $relative = $_.FullName.Substring($sourceDir.Length).TrimStart([char[]]"\/")
            $name = ($relative -replace "[\\/:]", "_") -replace "\.cpp$", ".o"
            if ($relative.Replace('\', '/') -eq 'src/combat/Actions.cpp') { $actionsObject }
            elseif ($relative.Replace('\', '/') -eq 'src/combat/CardInstance.cpp') { $cardObject }
            elseif ($relative.Replace('\', '/') -eq 'src/combat/BattleContext.cpp') { $battleObject }
            elseif ($relative.Replace('\', '/') -eq 'src/combat/CardManager.cpp') { $managerObject }
            else { Join-Path $objectsDir $name }
        }
)
$engineObjects += Join-Path $objectsDir "source_overlay_src_sim_search_BattleScumSearcher2.o"
$objectHashes = [ordered]@{}
foreach ($path in $engineObjects) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing pinned engine object: $path. Build the base simulator first."
    }
    $objectHashes[(Split-Path $path -Leaf)] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}
& $manifest.compiler.path @compileFlags -c $bridgeSource -o $objectPath
if ($LASTEXITCODE -ne 0) { throw "Card-selection bridge compilation failed" }
& $manifest.compiler.path $objectPath @engineObjects @($manifest.compiler.link_flags) -o $executable
if ($LASTEXITCODE -ne 0) { throw "Card-selection bridge link failed" }
$result = [ordered]@{
    schema_version = "card_selection_bridge_build_v1"
    mechanics = "corrected_v1"
    revision = $manifest.revision
    base_manifest_sha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    bridge_source_sha256 = (Get-FileHash -LiteralPath $bridgeSource -Algorithm SHA256).Hash.ToLowerInvariant()
    actions_upgrade_hand_sha256 = (Get-FileHash -LiteralPath $actionsPatch -Algorithm SHA256).Hash.ToLowerInvariant()
    cards_rage_cost_sha256 = (Get-FileHash -LiteralPath $ragePatch -Algorithm SHA256).Hash.ToLowerInvariant()
    card_mechanics_sha256 = (Get-FileHash -LiteralPath $mechanicsPatch -Algorithm SHA256).Hash.ToLowerInvariant()
    public_draw_memory_patch_sha256 = (Get-FileHash -LiteralPath $memoryPatch -Algorithm SHA256).Hash.ToLowerInvariant()
    public_draw_memory_header_sha256 = (Get-FileHash -LiteralPath $memoryHeader -Algorithm SHA256).Hash.ToLowerInvariant()
    bridge_sha256 = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant()
    engine_objects_sha256 = $objectHashes
    elapsed_seconds = ((Get-Date) - $started).TotalSeconds
}
[IO.File]::WriteAllText((Join-Path $outputDir "build_manifest.json"),
    ($result | ConvertTo-Json -Depth 6), (New-Object Text.UTF8Encoding($false)))
Write-Host "Card-selection bridge built: $executable"
Write-Host "Elapsed seconds: $($result.elapsed_seconds)"
