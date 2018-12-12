function createLine(term_icon, color, line_count){
    let term_cont = document.createElement("div");
    term_cont.setAttribute("class", "term-cont");

    let term_line = document.createElement("div");
    term_line.setAttribute("class", "term-line");

    let term_prompt = document.createElement("span");
    term_prompt.setAttribute("class", "term-prompt");
    term_prompt.innerHTML = term_icon;

    let term_cmd = document.createElement("span");
    term_cmd.setAttribute("class", "term-cmd current");
    term_cmd.id = line_count.toString();
    if(color != null) term_cmd.style.color = color;

    let term_caret = document.createElement("span");
    term_caret.setAttribute("class", "term-caret");
    term_caret.innerHTML="_";

    term_line.appendChild(term_prompt);
    term_line.appendChild(term_cmd);
    term_line.appendChild(term_caret);
    term_cont.appendChild(term_line);

    return term_cont;
}

class Sequence{

    constructor(term_icon, text, speed, delay, color){
        this.term_icon = term_icon;
        this.text = text;
        this.speed = speed;
        this.delay = delay;
        this.color = color;

        // keeps track of how much text has been written on screen
        this.writeLen = 0;
    }
}


class TypeWriter{

    constructor(sequences){
        this.sequences = sequences;
        this.seqCount = 0;
    }

    write(){
	    if(this.sequences == null || this.sequences.length == 0)
	        return;

        if(this.seqCount < this.sequences.length){
            let seq = this.sequences[this.seqCount];

            // create new line and append it to terminal
            let new_line = createLine(seq.term_icon, seq.color, this.seqCount);
            document.getElementById("term").appendChild(new_line);

            // get element where we write text too
            let term_text = document.getElementById(this.seqCount);

            // pass carrot in to be deleted later
            let cursor = document.getElementsByClassName("term-caret")[this.seqCount];

            // write sequence to new line
            this.writeSequence(seq, term_text, cursor, this);
        }
    }

    writeSequence(sequence, term_text, cursor, writer){

        if (sequence.writeLen < sequence.text.length){
            // update line we write text too
            term_text.innerHTML += sequence.text.charAt(sequence.writeLen);

            sequence.writeLen++;
            setTimeout(function(){writer.writeSequence(sequence, term_text, cursor, writer);}, sequence.speed);
        }
        else{
            // destroy cursor
            cursor.innerHTML=null;

            // pause and go to next sequence
            writer.seqCount++;
            setTimeout(function(){writer.write();}, sequence.delay);
        }
    }
}


document.addEventListener("DOMContentLoaded", function(){

    let seqs = [
        new Sequence("plogs@root$ ", "pip3 -U install plogs", 70, 500, null),
        new Sequence("plogs@root$ ", "python3", 100, 500, null),
        new Sequence(">>> ", "", 0, 300, null),
        new Sequence(">>> ", "from plogs import get_logger", 100, 300, null),
        new Sequence(">>> ", "logging = get_logger()", 100, 1000, null),
        new Sequence(">>> ", "", 0, 300, null),
        new Sequence(">>> ", "logging.success('hi')", 100, 100, null),
        new Sequence(">>> ", "hi", 0, 1000, "#00EE00"),
        new Sequence(">>> ", "", 0, 300, null),
        new Sequence(">>> ", "logging.critical('err')", 100, 100, null),
        new Sequence(">>> ", "err", 0, 100, "red")
    ];

    let writer = new TypeWriter(seqs);
    writer.write();
});
