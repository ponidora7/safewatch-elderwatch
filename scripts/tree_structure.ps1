# scripts/tree_structure.ps1
$repoRoot = Split-Path -Path $PSScriptRoot -Parent
$outputFile = Join-Path -Path $repoRoot -ChildPath "workspace_structure.txt"
$ignoreDirs = @('.git', 'venv', '__pycache__', '.vscode', '.idea', 'node_modules', '.next', 'out')

function Get-TreeStructure {
    param (
        [string]$Path,
        [int]$Level = 0
    )

    $indent = ""
    for ($i = 0; $i -lt $Level; $i++) {
        $indent += "|   "
    }

    $items = Get-ChildItem -Path $Path -Force -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -notin $ignoreDirs -and $_.Name -ne "workspace_structure.txt" -and $_.Name -ne "tree_structure.py" -and $_.Name -ne "tree_structure.ps1"
    } | Sort-Object -Property @{Expression={$_.PSIsContainer}; Descending=$true}, Name

    $count = $items.Count
    for ($i = 0; $i -lt $count; $i++) {
        $item = $items[$i]
        $isLast = ($i -eq ($count - 1))
        $branch = if ($isLast) { "\-- " } else { "|-- " }

        if ($item.PSIsContainer) {
            $line = "$($indent)$($branch)[DIR] $($item.Name)/"
            $line | Out-File -FilePath $outputFile -Append -Encoding ascii
            Get-TreeStructure -Path $item.FullName -Level ($Level + 1)
        } else {
            $line = "$($indent)$($branch)[FILE] $($item.Name)"
            $line | Out-File -FilePath $outputFile -Append -Encoding ascii
        }
    }
}

# Clear old structure
"Workspace: $(Split-Path -Path $repoRoot -Leaf)" | Out-File -FilePath $outputFile -Encoding ascii

Get-TreeStructure -Path $repoRoot
Write-Host "Success! Workspace structure saved to: $outputFile"
