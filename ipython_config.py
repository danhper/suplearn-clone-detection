# Configuration file for ipython.

#------------------------------------------------------------------------------
# InteractiveShellApp(Configurable) configuration
#------------------------------------------------------------------------------

## A Mixin for applications that start InteractiveShell instances.
#
#  Provides configurables for loading extensions and executing files as part of
#  configuring a Shell environment.
#
#  The following methods should be called by the :meth:`initialize` method of the
#  subclass:
#
#    - :meth:`init_path`
#    - :meth:`init_shell` (to be implemented by the subclass)
#    - :meth:`init_gui_pylab`
#    - :meth:`init_extensions`
#    - :meth:`init_code`

## Execute the given command string.
#c.InteractiveShellApp.code_to_run = ''

## Run the file referenced by the PYTHONSTARTUP environment variable at IPython
#  startup.
#c.InteractiveShellApp.exec_PYTHONSTARTUP = True

## List of files to run at IPython startup.
#c.InteractiveShellApp.exec_files = []

## lines of code to run at IPython startup.
c.InteractiveShellApp.exec_lines = ['%autoreload 2']

## A list of dotted module names of IPython extensions to load.
c.InteractiveShellApp.extensions = ['autoreload']

## A file to be run
# c.InteractiveShellApp.file_to_run = ''
c.InteractiveShellApp.exec_files = ['.ipython-startup.ipy']
