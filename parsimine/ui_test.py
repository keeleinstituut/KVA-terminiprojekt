import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import fitz # PyMuPDF
from parser_utils import calculate_header_and_footer_box

class PDFViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Viewer")
        self.geometry("900x900")

        # Create left and right frames
        left_frame_bg = None
        left_frame = tk.Frame(self, width=200, bg=left_frame_bg)
        left_frame.grid(row=0, column=0, padx=5, pady=5)

        right_frame = tk.Frame(self, width=800, bg='grey')
        right_frame.grid(row=0, column=1, columnspan=20, padx=5, pady=5)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)

        # LEFT FRAME
        # Create labels and text boxes for header and footer heights
        self.header_label = tk.Label(left_frame, text="Header Height:", bd=0, width=15,  bg=left_frame_bg)
        self.header_label.grid(row=1, column=2)
        self.header_height = tk.StringVar()
        self.header_height.set("60")
        self.header_entry = tk.Entry(left_frame, textvariable=self.header_height)
        self.header_entry.grid(row=1, column=3, pady=(20, 20))

        self.footer_label = tk.Label(left_frame, text="Footer Height:", bd=0, width=15,  bg=left_frame_bg)
        self.footer_label.grid(row=2, column=2)
        self.footer_height = tk.StringVar()
        self.footer_height.set("60")
        self.footer_entry = tk.Entry(left_frame, textvariable=self.footer_height)
        self.footer_entry.grid(row=2, column=3, pady=(0,20))

        self.term_start_page_label = tk.Label(left_frame, text="Glossary start:", bd=0, width=15, bg=left_frame_bg)
        self.term_start_page_label.grid(row=3, column=2)
        self.term_start_page_height = tk.StringVar()
        self.term_start_page_entry = tk.Entry(left_frame)
        self.term_start_page_entry.grid(row=3, column=3, pady=(20, 20))

        self.term_end_page_label = tk.Label(left_frame, text="Glossary end:", bd=0, width=15,  bg=left_frame_bg)
        self.term_end_page_label.grid(row=4, column=2)
        self.term_end_page_height = tk.StringVar()
        self.term_end_page_entry = tk.Entry(left_frame)
        self.term_end_page_entry.grid(row=4, column=3, pady=(0,20))

        # Create buttons for page navigation and file selection
        self.open_button = tk.Button(left_frame, text="Open PDF", command=self.open_pdf, width=15)
        self.open_button.grid(row=0, column=2, pady=(20, 0))

        self.save_and_parse_button = tk.Button(left_frame, text="Approve PDF", command=self.open_pdf, width=15)
        self.save_and_parse_button.grid(row=7, column=2)


        # RIGHT FRAME
        self.prev_button = tk.Button(right_frame, text="Previous", command=self.show_prev_page)
        self.prev_button.grid(row=7, column=0)

        self.next_button = tk.Button(right_frame, text="Next", command=self.show_next_page)
        self.next_button.grid(row=7, column=7)

        self.page_counter_label = tk.Label(right_frame, text="0 / 0")
        self.page_counter_label.grid(row=4, column=2)

        # Create canvas for displaying the PDF
        self.canvas = tk.Canvas(right_frame, bg='white')
        self.canvas.grid(row=5, rowspan=6, column=2, columnspan=4, sticky='nsew')


        # Bind the rerender function to the text boxes
        self.header_entry.bind("<Return>", self.rerender_page)
        self.footer_entry.bind("<Return>", self.rerender_page)

        # Configure the grid to expand the canvas
        #self.grid_columnconfigure(0, weight=3)
        #self.grid_rowconfigure(0, weight=3)

        # Initialize variables
        self.file_path = None
        self.doc = None
        self.num_pages = 0
        self.current_page = 0
        
    
    def open_pdf(self):
        # Open file dialog to select a PDF file
        self.file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if self.file_path:
            # Load the PDF
            self.doc = fitz.open(self.file_path)
            self.num_pages = len(self.doc)
            self.current_page = 0
            # Render the first page
            self.render_page(self.current_page)

    def render_page(self, page_num):
        # Clear canvas
        for widget in self.canvas.winfo_children():
            widget.destroy()

        # Render the specified page
        if self.doc:
            self.doc = fitz.open(self.file_path)
            self.page_counter_label.config(text=f'{page_num + 1} / {self.doc.page_count}')
            page = self.doc.load_page(page_num)
            header_box, footer_box = calculate_header_and_footer_box(page, header_height=int(self.header_height.get()), footer_height=int(self.footer_height.get()))
            # Create a new shape object for the page            

            shape = page.new_shape()
            shape.draw_rect(fitz.Rect(header_box))
            shape.draw_rect(fitz.Rect(footer_box))
            shape.finish(color=[0,0,1]) # Blue border, yellow fill
            shape.commit()

            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_tk = ImageTk.PhotoImage(img.resize(size=(int(pix.width/1.5), int(pix.height/1.5))))
            label = tk.Label(self.canvas, image=img_tk)
            label.pack(fill=tk.BOTH, expand=True)
            label.image = img_tk # Keep a reference to the image object

    def show_prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.render_page(self.current_page)

    def show_next_page(self):
        if self.current_page < self.num_pages - 1:
            self.current_page += 1
            self.render_page(self.current_page)

    def rerender_page(self, event):
        # Rerender the current page when the user changes the header or footer height
        self.render_page(self.current_page)

    def pdf_to_json(self):
        pass

if __name__ == "__main__":
    app = PDFViewer()
    app.mainloop()
