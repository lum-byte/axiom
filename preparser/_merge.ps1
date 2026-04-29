$base = "c:\Users\Kanishk\Desktop\Axiom\preparser"
$parts = @("_test_expansion.go")
$combined = ""
foreach ($p in $parts) {
    $path = Join-Path $base $p
    $content = Get-Content $path -Raw
    $content = $content -replace '(?m)^package preparser\s*\r?\n', ''
    $content = $content -replace '(?m)^import \(\s*\r?\n(?:\s*".*"\s*\r?\n)*\)\s*\r?\n', ''
    $combined += "`n$content"
}
Add-Content -Path (Join-Path $base "preparser_test.go") -Value $combined -NoNewline
foreach ($p in $parts) {
    Remove-Item (Join-Path $base $p) -Force
}
$count = (Get-Content (Join-Path $base "preparser_test.go")).Count
Write-Host "preparser_test.go now has $count lines"
