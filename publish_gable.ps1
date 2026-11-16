<#
.SYNOPSIS
  Publish only audited public Gable files to niko4244/gable in one GitHub commit.
.DESCRIPTION
  This workspace lives beneath an unrelated home-directory Git repository.
  Publishing uses the authenticated gh Git API with a strict path allowlist:
  never git add -A, and never touch that parent repository. Hidden benchmark
  cases, reference repairs, model artifacts and local receipts stay private.
  Omit -Push for a non-network inspection of the allowlist and content hashes.
#>
param(
    [switch]$Push,
    [string]$Message = "Update Gable public source"
)

$ErrorActionPreference = "Stop"
$repository = "niko4244/gable"
$root = $PSScriptRoot
Set-Location -LiteralPath $root

$publicFiles = @(
    ".gitignore",
    "README.md",
    "PUBLIC_RELEASE.md",
    "publish_gable.ps1",
    "rcl.py",
    "repo_rcl.py",
    "oracle_runner.py",
    "repo_case_runner.py",
    "compare_holdouts.py",
    "train.py",
    "source_pin.json",
    "C3_REPOSITORY_RCL_V1.md",
    "test_public_contracts.py",
    ".github/workflows/public-ci.yml"
)

$treeItems = [System.Collections.Generic.List[object]]::new()
foreach ($relative in $publicFiles) {
    $file = Join-Path $root $relative
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing allowlisted public source: $relative"
    }
    $value = [System.IO.File]::ReadAllText($file)
    if ($value -match '(?i)-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]{20,}|sk-proj-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}') {
        throw "Potential secret in public source: $relative"
    }
    $privateWindowsPrefix = "C:" + [char]92 + "Users" + [char]92 + "nik" + "ma" + [char]92
    $privateLinuxPrefix = "/home/" + "nik" + "ma/"
    if ($value.Contains($privateWindowsPrefix) -or $value.Contains($privateLinuxPrefix)) {
        throw "Machine-private absolute path in public source: $relative"
    }
    $sha = [System.Security.Cryptography.SHA256]::HashData(
        [System.Text.Encoding]::UTF8.GetBytes($value)
    )
    Write-Output ("PUBLIC {0} SHA256={1}" -f $relative, [Convert]::ToHexString($sha).ToLowerInvariant())
    $treeItems.Add(@{ path = $relative; mode = "100644"; type = "blob"; content = $value })
}

if (-not $Push) {
    Write-Output "DRY RUN: $($treeItems.Count) allowlisted files reviewed; no GitHub changes."
    return
}

function Invoke-GhJson {
    param([string]$Path, [string]$Method = "GET", [object]$Body = $null)
    $temp = $null
    try {
        if ($null -ne $Body) {
            $temp = [System.IO.Path]::GetTempFileName()
            $Body | ConvertTo-Json -Depth 20 -Compress |
                Set-Content -LiteralPath $temp -Encoding utf8 -NoNewline
            $raw = & gh api -X $Method $Path --input $temp
        } else {
            $raw = & gh api -X $Method $Path
        }
        if ($LASTEXITCODE -ne 0) {
            throw "GitHub API failed: $Method $Path"
        }
        return (($raw -join [Environment]::NewLine) | ConvertFrom-Json -Depth 30)
    } finally {
        if ($temp -and (Test-Path -LiteralPath $temp)) {
            Remove-Item -LiteralPath $temp -Force
        }
    }
}

# GitHub compare-and-swap. The old commit is the sole parent; nonfastforward
# updates are rejected by the remote. This never touches unrelated local Git.
$head = Invoke-GhJson "repos/$repository/git/ref/heads/main"
$oldCommit = $head.object.sha
$commit = Invoke-GhJson "repos/$repository/git/commits/$oldCommit"
$newTree = Invoke-GhJson "repos/$repository/git/trees" "POST" @{
    base_tree = $commit.tree.sha
    tree = @($treeItems.ToArray())
}
if ($newTree.sha -eq $commit.tree.sha) {
    Write-Output "No changes: public source matches latest Gable main."
    return
}
$newCommit = Invoke-GhJson "repos/$repository/git/commits" "POST" @{
    message = $Message
    tree = $newTree.sha
    parents = @($oldCommit)
}
$updated = Invoke-GhJson "repos/$repository/git/refs/heads/main" "PATCH" @{
    sha = $newCommit.sha
    force = $false
}
if ($updated.object.sha -ne $newCommit.sha) {
    throw "GitHub did not confirm the expected new Gable main commit"
}
Write-Output "PUBLISHED https://github.com/$repository/commit/$($newCommit.sha)"
