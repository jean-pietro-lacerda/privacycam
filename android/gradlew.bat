@echo off
setlocal
if defined JAVA_HOME (
    set "PRIVACYCAM_JAVA=%JAVA_HOME%\bin\java.exe"
) else (
    set "PRIVACYCAM_JAVA=java.exe"
)
"%PRIVACYCAM_JAVA%" -classpath "%~dp0gradle\wrapper\gradle-wrapper.jar" org.gradle.wrapper.GradleWrapperMain %*
exit /b %errorlevel%
