"""
Script to compile LaTeX document
"""

import os
import subprocess
import sys


def compile_latex(tex_file, output_dir=None):
    """Compile LaTeX document"""
    if output_dir is None:
        output_dir = os.path.dirname(tex_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Change to directory containing tex file
    tex_dir = os.path.dirname(os.path.abspath(tex_file))
    tex_filename = os.path.basename(tex_file)
    
    print(f"Compiling {tex_filename}...")
    print(f"Working directory: {tex_dir}")
    
    # Run pdflatex twice for cross-references
    for i in range(2):
        print(f"\nRun {i+1}...")
        result = subprocess.run(
            ['pdflatex', '-output-directory', output_dir, tex_filename],
            cwd=tex_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error compiling LaTeX:")
            print(result.stderr)
            return False
        
        # Print warnings if any
        if 'Warning' in result.stdout or 'Warning' in result.stderr:
            print("Warnings:")
            for line in result.stdout.split('\n'):
                if 'Warning' in line:
                    print(f"  {line}")
    
    print(f"\n✓ LaTeX compiled successfully!")
    pdf_file = os.path.join(output_dir, tex_filename.replace('.tex', '.pdf'))
    if os.path.exists(pdf_file):
        print(f"✓ PDF generated: {pdf_file}")
    
    return True


if __name__ == '__main__':
    if len(sys.argv) > 1:
        tex_file = sys.argv[1]
    else:
        # Default to report.tex in latex directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        tex_file = os.path.join(project_root, 'latex', 'report.tex')
    
    if not os.path.exists(tex_file):
        print(f"Error: LaTeX file not found: {tex_file}")
        sys.exit(1)
    
    output_dir = os.path.dirname(tex_file)
    success = compile_latex(tex_file, output_dir)
    
    if not success:
        sys.exit(1)

