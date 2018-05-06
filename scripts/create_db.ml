#!/usr/bin/ocamlrun ocaml

(* Installing dependencies:
  opam install core yojson sqlite3
*)

#use "topfind"
#thread
#require "core"
#require "yojson"
#require "sqlite3"


open Core


let default_root_dir =
  Filename.concat (Sys.getenv_exn "HOME") "workspaces/research/dataset/atcoder"
let root_dir = Option.value ~default:default_root_dir (Sys.getenv "DATA_ROOT")
let submissions_path = Filename.concat root_dir "submissions.json"
let asts_path = Filename.concat root_dir "asts/asts.jsonl"
let asts_names_path = Filename.concat root_dir "asts/asts.txt"
let db_path = Filename.concat root_dir "atcoder.db"

let sql_table_path =
  let project_root =
    Sys.argv.(0)
    |> Filename.realpath
    |> Filename.dirname
    |> Filename.dirname
  in
  Filename.concat project_root "sql/create_tables.sql"

let open_db () = Sqlite3.db_open db_path

let valid_languages = ["python"; "java"]

let asts =
  let asts_names = In_channel.read_lines asts_names_path in
  let asts = In_channel.read_lines asts_path in
  let mapped_asts = List.zip_exn asts_names asts in
  String.Map.of_alist_exn mapped_asts

let submissions =
  submissions_path
  |> Yojson.Safe.from_file
  |> Yojson.Safe.Util.to_list

let get_language_code language =
  List.find_exn valid_languages ~f:(fun v -> String.is_prefix ~prefix:v language)

let check_rc ?(expected=Sqlite3.Rc.OK) rc =
  if rc <> expected then
  failwith ("got invalid return code: " ^ (Sqlite3.Rc.to_string rc))

let insert_stmt =
  "INSERT INTO submissions (
    id,
    contest_id,
    contest_type,
    problem_id,
    problem_title,
    language,
    language_code,
    source_length,
    exec_time,
    tokens_count,
    ast,
    url
  ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"


let insert_submission stmt submission raw_ast =
  let open Yojson.Safe.Util in
  let int ?(from_str = false) key =
    let value = member key submission in
    let int_val = if from_str
                  then value |> to_string |> int_of_string
                  else to_int value
    in
    Sqlite3.Data.INT (Int64.of_int int_val)
  in
  let str key = Sqlite3.Data.TEXT (submission |> member key |> to_string) in

  let ast = Yojson.Safe.from_string raw_ast |> to_list in
  let tokens_count = Int64.of_int (List.length ast) in
  let language_code = get_language_code (submission |> member "language" |> to_string) in

  check_rc (Sqlite3.bind stmt 1 (int ~from_str:true "id"));
  check_rc (Sqlite3.bind stmt 2 (int "contest_id"));
  check_rc (Sqlite3.bind stmt 3 (str "contest_type"));
  check_rc (Sqlite3.bind stmt 4 (int "problem_id"));
  check_rc (Sqlite3.bind stmt 5 (str "problem_title"));
  check_rc (Sqlite3.bind stmt 6 (str "language"));
  check_rc (Sqlite3.bind stmt 7 (Sqlite3.Data.TEXT language_code));
  check_rc (Sqlite3.bind stmt 8 (int "source_length"));
  check_rc (Sqlite3.bind stmt 9 (int "exec_time"));
  check_rc (Sqlite3.bind stmt 10 (Sqlite3.Data.INT tokens_count));
  check_rc (Sqlite3.bind stmt 11 (Sqlite3.Data.TEXT raw_ast));
  check_rc (Sqlite3.bind stmt 12 (str "submission_url"));

  check_rc ~expected:Sqlite3.Rc.DONE (Sqlite3.step stmt);
  check_rc (Sqlite3.clear_bindings stmt);
  check_rc (Sqlite3.reset stmt)

let try_insert stmt submission =
  let open Yojson.Safe.Util in
  let file = submission |> member "file" |> to_string in
  match Map.find asts file with
  | Some ast -> insert_submission stmt submission ast
  | None -> print_endline ("ast not found for " ^ file)

let create_tables db =
  check_rc (Sqlite3.exec db (In_channel.read_all sql_table_path))

let wrap_in_transaction db ~f =
  check_rc (Sqlite3.exec db "BEGIN TRANSACTION");
  f ();
  check_rc (Sqlite3.exec db "END TRANSACTION")

let populate_db db =
  create_tables db;
  let stmt = Sqlite3.prepare db insert_stmt in
  let insert_all () = List.iter submissions ~f:(try_insert stmt) in
  wrap_in_transaction db ~f:insert_all;
  check_rc (Sqlite3.finalize stmt)

let () =
  let db = open_db () in
  populate_db db;
  ignore(Sqlite3.db_close db)
