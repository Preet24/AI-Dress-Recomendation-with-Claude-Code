# PreToolUse guard: deny any tool call that points at .env or .env.* (secrets).
# .env.example (the committed template) stays accessible.
# Only path-like fields and shell commands are checked, so writing docs that
# merely mention ".env" (e.g. CLAUDE.md) is not blocked.
$ErrorActionPreference = 'Stop'

$payload = [Console]::In.ReadToEnd() | ConvertFrom-Json
$inputs = $payload.tool_input
$fields = @('file_path', 'notebook_path', 'path', 'glob', 'command')
if ($payload.tool_name -eq 'Glob') { $fields += 'pattern' }

$targets = foreach ($name in $fields) {
    $value = $inputs.$name
    if ($value) { [string]$value }
}
$text = ($targets -join "`n") -replace '(?i)\.env\.example\b', ''

# ".env" not followed by a word character: matches .env, .env.local, prod.env;
# not .envrc, os.environ or env_file.
if ($text -match '(?i)\.env(?![\w-])') {
    @{
        hookSpecificOutput = @{
            hookEventName            = 'PreToolUse'
            permissionDecision       = 'deny'
            permissionDecisionReason = 'Access to .env files is blocked by the project hook (.claude/hooks/block-env.ps1); secrets stay with the user. .env.example is allowed.'
        }
    } | ConvertTo-Json -Depth 5 -Compress
}
exit 0
