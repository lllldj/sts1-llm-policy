[CmdletBinding()]
param(
    [string]$ConfigPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $projectRoot "configs\env\sts_lightspeed_build.json"
}

$resolvedConfigPath = (Resolve-Path -LiteralPath $ConfigPath).Path
$config = Get-Content -LiteralPath $resolvedConfigPath -Raw -Encoding UTF8 |
    ConvertFrom-Json

if ($config.schema_version -ne 1 -or $config.backend -ne "sts_lightspeed" -or $config.mechanics -ne "legacy_v1") {
    throw "Unsupported sts_lightspeed build configuration: $resolvedConfigPath"
}

function Resolve-ProjectPath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    return [IO.Path]::GetFullPath((Join-Path $projectRoot $RelativePath))
}

function Assert-PathWithin {
    param(
        [Parameter(Mandatory = $true)][string]$Candidate,
        [Parameter(Mandatory = $true)][string]$Parent,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $candidatePath = [IO.Path]::GetFullPath($Candidate).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $parentPath = [IO.Path]::GetFullPath($Parent).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $requiredPrefix = $parentPath + [IO.Path]::DirectorySeparatorChar

    if (-not $candidatePath.StartsWith(
        $requiredPrefix,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "$Label must remain inside $parentPath; got $candidatePath"
    }
}

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    Push-Location -LiteralPath $WorkingDirectory
    try {
        & $FilePath @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    if ($exitCode -ne 0) {
        throw "$FilePath failed with exit code $exitCode"
    }
}

function Invoke-Captured {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    Push-Location -LiteralPath $WorkingDirectory
    try {
        $output = @(& $FilePath @Arguments 2>&1 | ForEach-Object {
            $_.ToString()
        })
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    if ($exitCode -ne 0) {
        throw "$FilePath failed with exit code $exitCode`n$($output -join [Environment]::NewLine)"
    }

    return $output
}

$isolationRoot = Resolve-ProjectPath $config.isolation_root
$sourceDir = Resolve-ProjectPath $config.source_dir
$buildDir = Resolve-ProjectPath $config.build_dir
$manifestPath = Resolve-ProjectPath $config.manifest_path
$bridgeExe = Resolve-ProjectPath $config.artifacts.bridge
$compatHeader = Join-Path $projectRoot "tools\sts_lightspeed\compat.hpp"
$bridgeSource = Join-Path $projectRoot "tools\sts_lightspeed\decision_bridge.cpp"
$searchPatch = Join-Path $projectRoot "tools\sts_lightspeed\battle_scum_searcher2_bridge.patch"
$cardsPatch = Join-Path $projectRoot "tools\sts_lightspeed\cards_seeing_red.patch"

Assert-PathWithin -Candidate $isolationRoot -Parent $projectRoot -Label "isolation_root"
Assert-PathWithin -Candidate $sourceDir -Parent $isolationRoot -Label "source_dir"
Assert-PathWithin -Candidate $buildDir -Parent $isolationRoot -Label "build_dir"
Assert-PathWithin -Candidate $manifestPath -Parent $isolationRoot -Label "manifest_path"
Assert-PathWithin -Candidate $bridgeExe -Parent $isolationRoot -Label "bridge artifact"

if (-not (Test-Path -LiteralPath $bridgeSource -PathType Leaf)) {
    throw "Missing tracked decision bridge source: $bridgeSource"
}
if (-not (Test-Path -LiteralPath $searchPatch -PathType Leaf)) {
    throw "Missing tracked BattleScumSearcher2 bridge patch: $searchPatch"
}
if (-not (Test-Path -LiteralPath $cardsPatch -PathType Leaf)) {
    throw "Missing tracked Seeing Red card semantics patch: $cardsPatch"
}

$git = (Get-Command git -ErrorAction Stop).Source
$compiler = (Get-Command g++ -ErrorAction Stop).Source

New-Item -ItemType Directory -Path $isolationRoot -Force | Out-Null

$createdCheckout = $false
if (-not (Test-Path -LiteralPath $sourceDir)) {
    Write-Host "Cloning pinned sts_lightspeed source into isolated directory..."
    Invoke-External -FilePath $git -Arguments @(
        "clone",
        $config.upstream_url,
        $sourceDir
    ) -WorkingDirectory $isolationRoot
    $createdCheckout = $true
}

if (-not (Test-Path -LiteralPath (Join-Path $sourceDir ".git"))) {
    throw "Existing source_dir is not a Git checkout: $sourceDir"
}

$gitPrefix = @("-c", "safe.directory=$sourceDir", "-C", $sourceDir)
if (-not $createdCheckout) {
    $dirty = @(Invoke-Captured -FilePath $git -Arguments (
        $gitPrefix + @("status", "--porcelain", "--untracked-files=no")
    ) -WorkingDirectory $projectRoot)

    if ($dirty.Count -gt 0) {
        throw "Isolated upstream checkout has tracked modifications; refusing to overwrite them."
    }
}

& $git @($gitPrefix + @("cat-file", "-e", "$($config.revision)^{commit}"))
if ($LASTEXITCODE -ne 0) {
    Invoke-External -FilePath $git -Arguments (
        $gitPrefix + @("fetch", "origin", $config.revision)
    ) -WorkingDirectory $projectRoot
}

Invoke-External -FilePath $git -Arguments (
    $gitPrefix + @("checkout", "--detach", $config.revision)
) -WorkingDirectory $projectRoot
Invoke-External -FilePath $git -Arguments (
    $gitPrefix + @("submodule", "sync", "--recursive")
) -WorkingDirectory $projectRoot
Invoke-External -FilePath $git -Arguments (
    $gitPrefix + @("submodule", "update", "--init", "--recursive")
) -WorkingDirectory $projectRoot

$actualRevision = (
    Invoke-Captured -FilePath $git -Arguments (
        $gitPrefix + @("rev-parse", "HEAD")
    ) -WorkingDirectory $projectRoot |
        Select-Object -Last 1
).Trim()

if ($actualRevision -ne $config.revision) {
    throw "Revision mismatch: expected $($config.revision), got $actualRevision"
}

$actualSubmodules = [ordered]@{}
foreach ($property in $config.submodules.PSObject.Properties) {
    $submodulePath = $property.Name
    $expectedRevision = [string]$property.Value
    $resolvedSubmodule = Join-Path $sourceDir $submodulePath
    $actualSubmoduleRevision = (
        Invoke-Captured -FilePath $git -Arguments @(
            "-c",
            "safe.directory=$resolvedSubmodule",
            "-C",
            $resolvedSubmodule,
            "rev-parse",
            "HEAD"
        ) -WorkingDirectory $projectRoot |
            Select-Object -Last 1
    ).Trim()

    if ($actualSubmoduleRevision -ne $expectedRevision) {
        throw "Submodule $submodulePath mismatch: expected $expectedRevision, got $actualSubmoduleRevision"
    }
    $actualSubmodules[$submodulePath] = $actualSubmoduleRevision
}

$objectDir = Join-Path $buildDir "obj"
$overlayRoot = Join-Path $buildDir "source_overlay"
$overlayIncludeDir = Join-Path $overlayRoot "include"
$overlayHeader = Join-Path $overlayIncludeDir "sim\search\BattleScumSearcher2.h"
$overlaySource = Join-Path $overlayRoot "src\sim\search\BattleScumSearcher2.cpp"
$overlayCardsHeader = Join-Path $overlayIncludeDir "constants\Cards.h"
New-Item -ItemType Directory -Path $objectDir -Force | Out-Null
$binDir = Split-Path -Parent $bridgeExe
New-Item -ItemType Directory -Path $binDir -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $overlayHeader) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $overlaySource) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $overlayCardsHeader) -Force | Out-Null

Copy-Item -LiteralPath (
    Join-Path $sourceDir "include\sim\search\BattleScumSearcher2.h"
) -Destination $overlayHeader -Force
Copy-Item -LiteralPath (
    Join-Path $sourceDir "src\sim\search\BattleScumSearcher2.cpp"
) -Destination $overlaySource -Force
Copy-Item -LiteralPath (
    Join-Path $sourceDir "include\constants\Cards.h"
) -Destination $overlayCardsHeader -Force
$overlayRelative = $overlayRoot.Substring($projectRoot.Length).TrimStart(
    [char[]]"\/"
) -replace "\\", "/"
Invoke-External -FilePath $git -Arguments @(
    "apply",
    "--directory=$overlayRelative",
    $searchPatch
) -WorkingDirectory $projectRoot
Invoke-External -FilePath $git -Arguments @(
    "apply",
    "--directory=$overlayRelative",
    $cardsPatch
) -WorkingDirectory $projectRoot

$commonCompileArgs = @(
    "-std=c++17",
    "-O3",
    "-DNDEBUG",
    "-Wno-shift-count-overflow",
    "-include",
    $compatHeader,
    "-I",
    $overlayIncludeDir,
    "-I",
    (Join-Path $sourceDir "include"),
    "-I",
    (Join-Path $sourceDir "json\single_include")
)

$engineObjects = @()
$upstreamSearcherSource = Join-Path $sourceDir "src\sim\search\BattleScumSearcher2.cpp"
$engineSources = @(
    Get-ChildItem -LiteralPath (Join-Path $sourceDir "src") -Recurse -Filter "*.cpp" |
        Where-Object { $_.FullName -ne $upstreamSearcherSource } |
        Sort-Object FullName
)
$engineSources += Get-Item -LiteralPath $overlaySource

Write-Host "Compiling $($engineSources.Count) engine translation units..."
foreach ($source in $engineSources) {
    $relative = $source.FullName.Substring($sourceDir.Length).TrimStart(
        [char[]]"\/"
    )
    $objectName = ($relative -replace "[\\/:]", "_") -replace "\.cpp$", ".o"
    $objectPath = Join-Path $objectDir $objectName
    Invoke-External -FilePath $compiler -Arguments (
        $commonCompileArgs + @("-c", $source.FullName, "-o", $objectPath)
    ) -WorkingDirectory $projectRoot
    $engineObjects += $objectPath
}

$bridgeObject = Join-Path $objectDir "app_decision_bridge.o"
Invoke-External -FilePath $compiler -Arguments (
    $commonCompileArgs + @("-c", $bridgeSource, "-o", $bridgeObject)
) -WorkingDirectory $projectRoot

$linkArgs = @("-static", "-static-libgcc", "-static-libstdc++")
Invoke-External -FilePath $compiler -Arguments (
    @($bridgeObject) + $engineObjects + $linkArgs + @("-o", $bridgeExe)
) -WorkingDirectory $projectRoot

$compilerVersionOutput = @(Invoke-Captured -FilePath $compiler -Arguments @("--version") -WorkingDirectory $projectRoot)
$compilerVersion = ($compilerVersionOutput | Select-Object -First 1).Trim()

$manifest = [ordered]@{
    schema_version = 1
    mechanics = "legacy_v1"
    backend = "sts_lightspeed"
    transport = "native_process"
    upstream_url = $config.upstream_url
    revision = $actualRevision
    submodules = $actualSubmodules
    compiler = [ordered]@{
        path = $compiler
        version = $compilerVersion
        compile_flags = $commonCompileArgs
        link_flags = $linkArgs
    }
    source_overlays = [ordered]@{
        battle_scum_searcher2_bridge_patch = [ordered]@{
            path = "tools/sts_lightspeed/battle_scum_searcher2_bridge.patch"
            sha256 = (Get-FileHash -LiteralPath $searchPatch -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        cards_seeing_red_patch = [ordered]@{
            path = "tools/sts_lightspeed/cards_seeing_red.patch"
            sha256 = (Get-FileHash -LiteralPath $cardsPatch -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    artifacts = [ordered]@{
        bridge = [ordered]@{
            path = $config.artifacts.bridge
            sha256 = (Get-FileHash -LiteralPath $bridgeExe -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    built_at_utc = [DateTime]::UtcNow.ToString("o")
}

$manifestJson = $manifest | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText(
    $manifestPath,
    $manifestJson + [Environment]::NewLine,
    [Text.UTF8Encoding]::new($false)
)

Write-Host "sts_lightspeed build complete."
Write-Host "  source:   $sourceDir"
Write-Host "  bridge:   $bridgeExe"
Write-Host "  manifest: $manifestPath"
